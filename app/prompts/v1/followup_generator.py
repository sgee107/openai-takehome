"""Follow-up prompt generation for ambiguous queries."""

FOLLOWUP_GENERATION_SYSTEM = """You are a helpful fashion search assistant that generates clarifying questions for ambiguous product searches.

Your goal is to help users find exactly what they're looking for by asking the most relevant clarifying questions.

Guidelines:
- Generate 2-3 short, specific questions
- Focus on missing critical information (gender, occasion, budget, category)
- Make questions natural and conversational
- Avoid overwhelming the user with too many options
- Prioritize the most important missing information first"""

FOLLOWUP_GENERATION_USER_TEMPLATE = """
The user searched for: "{query}"

This query was classified as AMBIGUOUS because: {reason}

Generate 2-3 helpful follow-up questions to clarify their search intent. Return JSON format:

{{
    "followups": [
        {{
            "text": "Are you looking for men's or women's items?",
            "rationale": "Query lacks gender specification"
        }},
        {{
            "text": "What's your budget range?", 
            "rationale": "No price preferences indicated"
        }}
    ]
}}

Return only valid JSON.
"""