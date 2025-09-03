"""Intent classification prompts for complexity scoring and query parsing."""

INTENT_CLASSIFICATION_SYSTEM = """You are a fashion product search classifier. Analyze user queries and return structured JSON.

Classification Rules:
1 (DIRECT): Specific product searches - exact brand+model, SKUs, quoted product names, specific sizes like "32x34"
2 (FILTERED): Category searches with constraints - mentions colors, price ranges, categories, brands
3 (AMBIGUOUS): Broad searches - general terms like "work clothes", "outfit", style descriptions

Always extract structured information when present."""

INTENT_CLASSIFICATION_USER_TEMPLATE = """
Analyze this fashion search query and return a JSON response with:

{{
    "complexity": 1 | 2 | 3,
    "reason": "brief explanation of classification",
    "parsed": {{
        "category": "clothing category if mentioned (e.g., dress, shoes, shirt)",
        "brand": "brand name if mentioned", 
        "price_min": number or null,
        "price_max": number or null
    }}
}}

Query: "{query}"

Return only valid JSON.
"""
