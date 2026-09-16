"""
General Kleinanzeigen Agent - Configurable Search & Analysis Tool

Startaufruf über das Terminal:
python kleinanzeigen_agent_general.py --config configs/tv_defekt.json
"""

import argparse
import asyncio
import os
import sys
import re
import json
from pathlib import Path
import httpx
from typing import Literal, Optional
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Bypass MCP server and use the client directly
sys.path.insert(0, str(Path(__file__).parent / "ebay-kleinanzeigen-api-mcp" / "src") if '__file__' in globals() else "ebay-kleinanzeigen-api-mcp/src")
from kleinanzeigen_mcp.client import KleinanzeigenClient

# Load environment variables
load_dotenv()

# =============================================================================
# DATA MODELS
# =============================================================================

class Bewertung(BaseModel):
    listing_id: str
    titel: str
    preis_eur: float | None
    empfehlung: Literal["Ja", "Nein", "Vorsicht"]
    begruendung: str

class BatchBewertung(BaseModel):
    bewertungen: list[Bewertung]

# =============================================================================
# CORE CONFIG & CLIENTS
# =============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set. Please create a .env file with your API key.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Stabile Modell-Kette
LLM_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash"
]
LLM_RETRIES = 2
LLM_DELAY = 2

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def build_image_parts(image_urls, listing_id="Unbekannt", max_images=None):
    """Lädt ALLE Bilder eines Inserats herunter und gibt die Anzahl im Log aus."""
    urls = image_urls[:max_images] if max_images else image_urls
    if not urls:
        return []

    async with httpx.AsyncClient(timeout=10) as client:
        async def fetch(url):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                mime = resp.headers.get("content-type", "image/jpeg")
                return types.Part.from_bytes(data=resp.content, mime_type=mime)
            except Exception as e:
                print(f"   [!] Bild-Download fehlgeschlagen ({url}): {e}")
                return None

        tasks = [fetch(url) for url in urls]
        results = await asyncio.gather(*tasks)
        valid_parts = [p for p in results if p is not None]
        
        # TRANS TRANSPARENZ-LOG: Zeigt Erfolg pro Inserat
        print(f"   [+] ID {listing_id}: {len(valid_parts)}/{len(urls)} Bilder erfolgreich geladen")
        return valid_parts


async def evaluate_batch(listings_batch, search_criteria) -> list[Bewertung]:
    """Analysiert Inserate inkl. ALLER zugehöriger Bilder im gleichen Kontext-Block."""
    contents = []
    total_images_in_batch = 0
    
    prompt_header = f"""
Du bist mein Einkaufs-Assistent. Prüfe die folgenden {len(listings_batch)} Inserate.
Jedes Inserat ist klar durch seine ID getrennt. Die Bilder eines Inserats folgen DIREKT unter dessen Textbeschreibung.

SUCHKRITERIEN:
{search_criteria}

INSERATE UND BILDER:
"""
    contents.append(prompt_header)
    
    for idx, listing in enumerate(listings_batch, 1):
        listing_id = listing.adid if hasattr(listing, 'adid') else getattr(listing, 'id', 'Unbekannt')
        
        item_text = f"\n=========================================\n"
        item_text += f"INSERAT {idx} (ID: {listing_id})\n"
        item_text += f"Titel: {listing.title}\n"
        item_text += f"Preis: {getattr(listing, 'price', 'Unbekannt')}\n"
        item_text += f"Beschreibung: {getattr(listing, 'description', '')[:500]}\n"
        contents.append(item_text)
        
        if hasattr(listing, 'images') and listing.images:
            # listing_id zur Kontrollausgabe mitgeben
            img_parts = await build_image_parts(listing.images, listing_id=listing_id, max_images=None)
            if img_parts:
                contents.append(f"\n[Bilder zu Inserat ID {listing_id} ({len(img_parts)} Stück)]:\n")
                contents.extend(img_parts)
                total_images_in_batch += len(img_parts)

    print(f"-> Sendung an Gemini: {len(listings_batch)} Inserate mit INSGESAMT {total_images_in_batch} Bildern.")

    contents.append("""
\n=========================================\n
AUFGABE:
Analysiere den Text UND die direkt zugeordneten Bilder jedes Inserats.
Gib für JEDES Inserat ein Bewertungsobjekt im JSON-Array zurück.
Achte darauf, dass die `listing_id` exakt mit der ID des jeweiligen Inserats übereinstimmt.
""")

    for model in LLM_MODELS:
        try:
            response = await gemini_client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchBewertung,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    temperature=0.2
                )
            )
            parsed: BatchBewertung = response.parsed
            
            for b in parsed.bewertungen:
                matching = next((l for l in listings_batch if (hasattr(l, 'adid') and l.adid == b.listing_id) or (hasattr(l, 'id') and l.id == b.listing_id)), None)
                if matching and hasattr(matching, 'price') and matching.price:
                    match = re.search(r'(\d+)', str(matching.price).replace('.', ''))
                    if match:
                        b.preis_eur = float(match.group(1))
                        
            return parsed.bewertungen
        except Exception as e:
            print(f"Warnung: Batch mit Modell {model} fehlgeschlagen: {e}")
            await asyncio.sleep(3)
            
    return []

# =============================================================================
# MAIN LOGIC
# =============================================================================

async def main():
    # 1. Config laden via Command Line Argument
    parser = argparse.ArgumentParser(description="Universal Kleinanzeigen AI Agent")
    parser.add_argument("--config", type=str, required=True, help="Pfad zur JSON-Konfigurationsdatei")
    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Fehler beim Laden der Konfigurationsdatei '{args.config}': {e}")
        return

    # Variablen aus der Config extrahieren (mit sicheren Fallbacks)
    search_cfg = config.get("search", {})
    s_query = search_cfg.get("query", "")
    s_location = search_cfg.get("location", "")
    s_radius = search_cfg.get("radius_km", 0)
    s_min_price = search_cfg.get("min_price", None)
    s_max_price = search_cfg.get("max_price", None)
    s_pages = search_cfg.get("max_pages", 1)
    
    criteria = config.get("criteria", "")
    manual_listings = config.get("manual_listings", [])
    output_file = config.get("output_file", "ergebnis_analyse.json")

    print("=" * 60)
    print(f"KLEINANZEIGEN AGENT - Config: {os.path.basename(args.config)}")
    print("=" * 60)
    print(f"Suche nach '{s_query}' in {s_location} (+{s_radius} km)")
    print(f"Preisrahmen: {s_min_price if s_min_price is not None else '0'}€ bis {s_max_price if s_max_price is not None else '∞'}€")
    print("-" * 60)

    # 2. Direkte API-Nutzung inkl. Preisgrenzen
    async with KleinanzeigenClient() as client:
        listings = []
        
        try:
            search_results = await client.search_listings(
                query=s_query,
                location=s_location,
                radius=s_radius,
                min_price=s_min_price, 
                max_price=s_max_price,
                page_count=s_pages
            )
            listings.extend(search_results)
            print(f"Suche abgeschlossen! {len(search_results)} Ergebnisse gefunden.")
        except Exception as e:
            print(f"Fehler bei der Suche: {e}")
            
        # Manuelle Inserate einfügen
        if manual_listings:
            print(f"\nLade {len(manual_listings)} manuelle Inserate...")
            for adid in manual_listings:
                try:
                    details = await client.get_listing_details(adid)
                    listings.append(details)
                except Exception as e:
                    print(f"Fehler bei manuellem Inserat {adid}: {e}")
        
        # Alle Bilder aus den Detailseiten laden
        print(f"\nLade vollständige Details & Galerien für {len(listings)} Inserate...")
        detailed_listings = []
        for idx, l in enumerate(listings, 1):
            try:
                d = await client.get_listing_details(l.adid)
                # Falls Detailseite keine Bilder liefert, Vorschaubild als Fallback nutzen
                if not d.images and hasattr(l, 'images') and l.images:
                    d.images = l.images
                detailed_listings.append(d)
            except Exception as e:
                detailed_listings.append(l)
        
        listings = detailed_listings
                    
        if not listings:
            print("Keine Inserate zum Analysieren gefunden. Prüfe deine Suchparameter/Preisgrenzen.")
            return

        # =====================================================================
        # BATCH-ANALYSE (15 Inserate pro Gemini-Request)
        # =====================================================================
        print("\nStarte KI-Batch-Analyse mit Bildverarbeitung...")
        BATCH_SIZE = 15
        all_bewertungen = []

        # Inserate in 15er-Blöcke aufteilen
        batches = [listings[i:i + BATCH_SIZE] for i in range(0, len(listings), BATCH_SIZE)]

        for idx, batch in enumerate(batches, 1):
            print(f"Verarbeite Batch {idx}/{len(batches)} ({len(batch)} Inserate)...")
            results = await evaluate_batch(batch, criteria)
            all_bewertungen.extend(results)

            # 6 Sekunden Pause zwischen Batches (Schutz vor 429 Rate Limits)
            if idx < len(batches):
                await asyncio.sleep(6)

        bewertungen = all_bewertungen
        
        # 3. Strukturierte Ergebnisse sortieren: "Ja" -> "Vorsicht" -> "Nein", dann nach Preis
        order = {"Ja": 0, "Vorsicht": 1, "Nein": 2}
        bewertungen.sort(key=lambda x: (order.get(x.empfehlung, 3), x.preis_eur or float('inf')))

        # 4. In der Konsole anzeigen
        print("\n" + "=" * 60)
        print("FINALE ERGEBNISSE (sortiert)")
        print("=" * 60)
        
        for b in bewertungen:
            preis_str = f"{b.preis_eur} €" if b.preis_eur is not None else "Preis unbekannt"
            print(f"[{b.empfehlung.upper()}] {b.titel} ({preis_str}) | ID: {b.listing_id}")
            print(f"Begründung: {b.begruendung}\n")

        # 5. Als maschinenlesbares JSON abspeichern
        with open(output_file, "w", encoding="utf-8") as f:
            json_data = [b.model_dump() for b in bewertungen]
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            
        print(f"Ergebnisse als strukturiertes JSON in '{output_file}' gespeichert.")

if __name__ == "__main__":
    asyncio.run(main())