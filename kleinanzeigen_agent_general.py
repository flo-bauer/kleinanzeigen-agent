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

async def build_image_parts(image_urls, max_images=3):
    """Lädt Bilder asynchron herunter und bereitet sie als Bytes für Gemini vor."""
    parts = []
    async with httpx.AsyncClient(timeout=10) as client:
        for url in image_urls[:max_images]:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                mime = resp.headers.get("content-type", "image/jpeg")
                parts.append(types.Part.from_bytes(data=resp.content, mime_type=mime))
            except Exception as e:
                print(f"Bild konnte nicht geladen werden ({url}): {e}")
    return parts

async def evaluate_listing(listing, search_criteria, semaphore) -> Optional[Bewertung]:
    """Bewertet ein einzelnes Inserat parallel mit Gemini."""
    async with semaphore:
        listing_id = listing.adid if hasattr(listing, 'adid') else getattr(listing, 'id', 'Unbekannt')
        
        # Versuche Preis in einen Float zu parsen für saubere Sortierung
        preis_eur = None
        if hasattr(listing, 'price') and listing.price:
            match = re.search(r'(\d+)', str(listing.price).replace('.', ''))
            if match:
                preis_eur = float(match.group(1))
                
        prompt_text = f"""
Du bist mein Einkaufs-Assistent für spezialisierte Produktsuchen.

SUCHKRITERIEN DES NUTZERS:
{search_criteria}

INSERAT ZUR PRÜFUNG:
Titel: {listing.title}
Preis: {getattr(listing, 'price', 'Unbekannt')}
Beschreibung: {getattr(listing, 'description', '')[:1000]} 

AUFGABE:
1. Analysiere das Angebot auf Passgenauigkeit zu den Kriterien.
2. Gib eine klare KAUFEMPFEHLUNG (Ja / Nein / Vorsicht) ab.
3. Begründe kurz, warum der Artikel passt oder nicht (bezogen auf zwingende Kriterien).
4. Wenn Bilder angehängt sind: Nutze sie zur Beurteilung (Gebrauchsspuren, sichtbare Ausstattungsmerkmale, Defekte).
"""
        
        image_parts = []
        if hasattr(listing, 'images') and listing.images:
            image_parts = await build_image_parts(listing.images, max_images=3)
            
        contents = [prompt_text, *image_parts]
        
        for model in LLM_MODELS:
            for attempt in range(LLM_RETRIES):
                try:
                    response = await gemini_client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=Bewertung,
                            temperature=0.2
                        )
                    )
                    bewertung = response.parsed
                    # Absichern, dass ID und Preis immer im Output landen
                    bewertung.listing_id = listing_id
                    bewertung.preis_eur = preis_eur
                    return bewertung
                except Exception as e:
                    print(f"Warnung: Modell {model} (Versuch {attempt+1}) fehlgeschlagen bei ID {listing_id}: {e}")
                    await asyncio.sleep(LLM_DELAY)
        
        print(f"Fehler: Alle Versuche für '{listing.title}' fehlgeschlagen. Überspringe...")
        return None

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
                    
        if not listings:
            print("Keine Inserate zum Analysieren gefunden. Prüfe deine Suchparameter/Preisgrenzen.")
            return

        print("\nStarte KI-Analyse mit Bildverarbeitung (Parallel)...")
        # Semaphore begrenzt parallele Requests (z.B. max 5 gleichzeitig)
        semaphore = asyncio.Semaphore(5) 
        
        tasks = [evaluate_listing(listing, criteria, semaphore) for listing in listings]
        results = await asyncio.gather(*tasks)
        
        # Leere Ergebnisse herausfiltern
        bewertungen = [res for res in results if res is not None]
        
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