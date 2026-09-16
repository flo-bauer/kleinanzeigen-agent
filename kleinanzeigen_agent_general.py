"""
General Kleinanzeigen Agent - Configurable Search & Analysis Tool

Usage via terminal:
python kleinanzeigen_agent_general.py --config configs/{config}.json
"""

import argparse
import asyncio
import os
import sys
import re
import json
import logging
from pathlib import Path
import httpx
from typing import Literal, Optional
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Suppress INFO and WARNING messages from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("kleinanzeigen_mcp.client").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# Bypass MCP server and use the client directly
sys.path.insert(0, str(Path(__file__).parent / "ebay-kleinanzeigen-api-mcp" / "src") if '__file__' in globals() else "ebay-kleinanzeigen-api-mcp/src")
from kleinanzeigen_mcp.client import KleinanzeigenClient

# Load environment variables
load_dotenv()

# =============================================================================
# DATA MODELS
# =============================================================================

class Evaluation(BaseModel):
    """Evaluation result for a single listing."""
    listing_id: str
    title: str
    price_eur: float | None
    recommendation: Literal["Yes", "No", "Caution"]
    reasoning: str

class BatchEvaluation(BaseModel):
    """Container for multiple listing evaluations."""
    evaluations: list[Evaluation]

# =============================================================================
# CORE CONFIG & CLIENTS
# =============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set. Please create a .env file with your API key.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Stable model chain (fallback order)
LLM_MODELS = [
    # "gemini-3.8-flash",
    # "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash"
]
LLM_RETRIES = 2
LLM_DELAY = 2

# Concurrency for fetching full listing details (parallel browser contexts)
DETAIL_FETCH_CONCURRENCY = 5

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def build_image_parts(image_urls: list[str], listing_id: str = "Unknown", max_images: int | None = None) -> list[types.Part]:
    """
    Download images for a listing and return them as Gemini Parts.
    
    Args:
        image_urls: List of image URLs to download.
        listing_id: Listing ID for logging purposes.
        max_images: Maximum number of images to download (None = all).
    
    Returns:
        List of Gemini Part objects containing image data.
    """
    urls = image_urls[:max_images] if max_images else image_urls
    if not urls:
        return []

    async with httpx.AsyncClient(timeout=10) as client:
        async def fetch(url: str) -> types.Part | None:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                mime = resp.headers.get("content-type", "image/jpeg")
                return types.Part.from_bytes(data=resp.content, mime_type=mime)
            except Exception as e:
                print(f"   [!] Image download failed ({url}): {e}")
                return None

        tasks = [fetch(url) for url in urls]
        results = await asyncio.gather(*tasks)
        valid_parts = [p for p in results if p is not None]

        print(f"   [+] ID {listing_id}: {len(valid_parts)}/{len(urls)} images loaded successfully")
        return valid_parts


async def fetch_full_details(
    client: KleinanzeigenClient, 
    summaries: list, 
    concurrency: int = DETAIL_FETCH_CONCURRENCY
) -> list:
    """
    Enrich search results (ListingSummary) with full detail pages in parallel with progress bar.
    
    Args:
        client: KleinanzeigenClient instance.
        summaries: List of ListingSummary objects from search.
        concurrency: Maximum concurrent browser contexts.
    
    Returns:
        List of enriched ListingDetails objects (falls back to summary on error).
    """
    semaphore = asyncio.Semaphore(concurrency)
    
    # Initialize progress bar
    pbar = tqdm(total=len(summaries), desc="Loading listing details", unit=" listings", ncols=80)

    async def fetch_one(listing):
        listing_id = getattr(listing, "adid", None)
        if not listing_id:
            pbar.update(1)
            return listing
        async with semaphore:
            try:
                full = await client.get_listing_details(listing_id)
                
                # Fallback to search result data
                if full.id == "[ERROR] ID not found" or not full.id:
                    full.id = listing_id
                
                if full.title == "[ERROR] Title not found" or not full.title:
                    full.title = getattr(listing, "title", "Title unknown")
                    
                if not full.price and hasattr(listing, "price"):
                    full.price = listing.price
                    
                if not full.images and hasattr(listing, "images"):
                    full.images = listing.images
                
                pbar.update(1)
                return full
            except Exception as e:
                tqdm.write(f"   [!] Failed to load details for {listing_id}: {e}")
                pbar.update(1)
                return listing

    results = await asyncio.gather(*(fetch_one(l) for l in summaries))
    pbar.close()
    return list(results)


async def evaluate_batch(listings_batch: list, search_criteria: str) -> list[Evaluation]:
    """
    Analyze a batch of listings including ALL associated images in the same context block.
    
    Args:
        listings_batch: List of ListingDetails objects to evaluate.
        search_criteria: User-defined search criteria for the AI.
    
    Returns:
        List of Evaluation objects.
    """
    contents = []
    total_images_in_batch = 0

    prompt_header = f"""
You are my shopping assistant. Review the following {len(listings_batch)} listings.
Each listing is clearly separated by its ID. The images for a listing follow DIRECTLY under its text description.

SEARCH CRITERIA:
{search_criteria}

LISTINGS AND IMAGES:
"""
    contents.append(prompt_header)

    for idx, listing in enumerate(listings_batch, 1):
        listing_id = listing.adid if hasattr(listing, 'adid') else getattr(listing, 'id', 'Unknown')

        item_text = f"\n=========================================\n"
        item_text += f"LISTING {idx} (ID: {listing_id})\n"
        item_text += f"Title: {listing.title}\n"
        item_text += f"Price: {getattr(listing, 'price', 'Unknown')}\n"
        item_text += f"Description: {getattr(listing, 'description', '')[:500]}\n"
        contents.append(item_text)

        if hasattr(listing, 'images') and listing.images:
            img_parts = await build_image_parts(listing.images, listing_id=listing_id, max_images=None)
            if img_parts:
                contents.append(f"\n[Images for Listing ID {listing_id} ({len(img_parts)} images)]:\n")
                contents.extend(img_parts)
                total_images_in_batch += len(img_parts)

    print(f"-> Sending to Gemini: {len(listings_batch)} listings with TOTAL {total_images_in_batch} images.")

    contents.append("""
\n=========================================\n
TASK:
Analyze the text AND the directly assigned images for each listing.
Return an evaluation object in the JSON array for EACH listing.
Ensure the `listing_id` exactly matches the ID of the respective listing.
""")

    for model in LLM_MODELS:
        for attempt in range(1, LLM_RETRIES + 1):
            try:
                response = await gemini_client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BatchEvaluation,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                        temperature=0.2
                    )
                )
                parsed: BatchEvaluation = response.parsed

                thoughts = getattr(response.usage_metadata, "thoughts_token_count", None)
                if thoughts:
                    print(f"   (Model {model}: {thoughts} thinking tokens used)")

                # Prefer price from our own (already parsed) raw data
                # rather than trusting the number returned by Gemini.
                for eval_item in parsed.evaluations:
                    matching = next(
                        (l for l in listings_batch 
                         if (hasattr(l, 'adid') and l.adid == eval_item.listing_id) 
                         or (hasattr(l, 'id') and l.id == eval_item.listing_id)), 
                        None
                    )
                    if matching and hasattr(matching, 'price') and matching.price:
                        match = re.search(r'(\d+)', str(matching.price).replace('.', ''))
                        if match:
                            eval_item.price_eur = float(match.group(1))

                return parsed.evaluations
            except Exception as e:
                # Strip JSON block, keep only e.g. "503 UNAVAILABLE"
                short_error = str(e).split('{')[0].strip()
                if not short_error:
                    short_error = str(e)[:60]  # Fallback
                
                print(f"   [!] {model} (Attempt {attempt}/{LLM_RETRIES}) failed: {short_error}")
                
                if attempt < LLM_RETRIES:
                    await asyncio.sleep(LLM_DELAY)

    return []

# =============================================================================
# MAIN LOGIC
# =============================================================================

async def main():
    # Load config via command line argument
    parser = argparse.ArgumentParser(description="Universal Kleinanzeigen AI Agent")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file '{args.config}': {e}")
        return

    search_cfg = config.get("search", {})
    s_query = search_cfg.get("query", "")
    s_location = search_cfg.get("location", "")
    s_radius = search_cfg.get("radius_km", 0)
    s_min_price = search_cfg.get("min_price", None)
    s_max_price = search_cfg.get("max_price", None)
    s_pages = search_cfg.get("max_pages", 1)

    criteria = config.get("criteria", "")
    manual_listings = config.get("manual_listings", [])
    output_file = config.get("output_file", "analysis_result.json")

    print("=" * 60)
    print(f"KLEINANZEIGEN AGENT - Config: {os.path.basename(args.config)}")
    print("=" * 60)
    print(f"Searching for '{s_query}' in {s_location} (+{s_radius} km)")
    print(f"Price range: {s_min_price if s_min_price is not None else '0'}€ to {s_max_price if s_max_price is not None else '∞'}€")
    print("-" * 60)

    async with KleinanzeigenClient() as client:
        try:
            search_results = await client.search_listings(
                query=s_query,
                location=s_location,
                radius=s_radius,
                min_price=s_min_price,
                max_price=s_max_price,
                page_count=s_pages,
            )
            print(f"Search completed! {len(search_results)} results found.")
        except Exception as e:
            print(f"Search error: {e}")
            search_results = []

        manual_details = []
        if manual_listings:
            print(f"\nLoading {len(manual_listings)} manual listings...")
            for adid in manual_listings:
                try:
                    manual_details.append(await client.get_listing_details(adid))
                except Exception as e:
                    print(f"Error loading manual listing {adid}: {e}")

        if not search_results and not manual_details:
            print("No listings found to analyze. Check your search parameters/price limits.")
            return

        # Load full details + image galleries for search results (parallelized)
        print(f"\nLoading full details & galleries for {len(search_results)} search results (concurrency: {DETAIL_FETCH_CONCURRENCY})...")
        enriched = await fetch_full_details(client, search_results)
        listings = enriched + manual_details

        # =====================================================================
        # BATCH ANALYSIS
        # =====================================================================
        print("\nStarting AI batch analysis with image processing...")
        BATCH_SIZE = 15
        all_evaluations = []

        batches = [listings[i:i + BATCH_SIZE] for i in range(0, len(listings), BATCH_SIZE)]

        for idx, batch in enumerate(batches, 1):
            print(f"Processing batch {idx}/{len(batches)} ({len(batch)} listings)...")
            results = await evaluate_batch(batch, criteria)
            all_evaluations.extend(results)

            if idx < len(batches):
                await asyncio.sleep(6)  # Protect against 429 rate limits between batches

        evaluations = all_evaluations

        # Sort: "Yes" -> "Caution" -> "No", then by price ascending.
        # Important: 0.0 € is a valid (lowest!) price, not "unknown" -
        # `x.price_eur or float('inf')` would incorrectly sort 0€ listings to the end
        # because 0.0 is falsy in Python. Therefore explicitly check for None.
        order = {"Yes": 0, "Caution": 1, "No": 2}
        evaluations.sort(key=lambda x: (order.get(x.recommendation, 3), x.price_eur if x.price_eur is not None else float('inf')))

        print("\n" + "=" * 60)
        print("FINAL RESULTS (sorted)")
        print("=" * 60)

        for e in evaluations:
            price_str = f"{e.price_eur} €" if e.price_eur is not None else "Price unknown"
            print(f"[{e.recommendation.upper()}] {e.title} ({price_str}) | ID: {e.listing_id}")
            print(f"Reasoning: {e.reasoning}\n")

        with open(output_file, "w", encoding="utf-8") as f:
            json_data = [e.model_dump() for e in evaluations]
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved as structured JSON to '{output_file}'.")

if __name__ == "__main__":
    asyncio.run(main())