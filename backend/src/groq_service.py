"""
@file groq_service.py
@description Groq integration for precise text extraction.

This service uses Groq (with OpenAI-compatible API) for the "Highlight with AI" 
feature to find exact verbatim phrases in documents that support AI-generated answers.

Why Groq for highlights?
- Fast inference with OpenAI compatibility
- JSON mode for structured outputs
- Good at finding exact phrases in source documents
- Fallback to Gemini if Groq fails

Required environment variables:
- GROQ_API_KEY: For primary highlight generation
- GEMINI_API_KEY: For fallback (optional)
"""

import os
import json
from openai import OpenAI
from typing import List
import google.generativeai as genai


async def get_relevance_highlights(answer: str, document_content: str) -> List[str]:
    """
    Triggers a secondary semantic analysis pass to find supporting text.
    
    This function powers the "Highlight with AI" feature. When a user views a
    source document, they can click to highlight which specific phrases support
    the chatbot's answer.
    
    Process:
    1. Receives the AI's previous answer and full document content
    2. Asks Groq to find 5-8 short exact phrases in the document
    3. Returns only verbatim strings (no paraphrasing)
    4. Frontend injects <mark> tags around these phrases
    
    Design decisions:
    - Uses Groq (primary) with Gemini fallback
    - Uses JSON mode for structured outputs
    - Instructs AI to avoid markdown characters for better matching
    - Returns short phrases (3-6 words) for precise highlighting
    
    Args:
        answer: The AI's previous answer (what we're verifying)
        document_content: Full source document to search
        
    Returns:
        List of verbatim strings to highlight (e.g., ["25 vacation days", "full-time employees"])
        
    Raises:
        Does not raise; returns empty list on error
    """
    # Try Groq first
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        # Configure Groq client with OpenAI-compatible API
        groq_client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Craft prompt with strict rules for verbatim extraction
        system_prompt = "You are a precision text extraction engine. Return ONLY valid JSON objects."
        
        user_prompt = f"""Find 5-8 short, key phrases (3-6 words each) in the DOCUMENT that specifically support the claims in the ANSWER.
            
STRICT RULES:
1. Return a JSON object with a "highlights" key containing an array of strings.
2. Each string MUST be a LITERALLY EXACT VERBATIM substring from the DOCUMENT.
3. Choose phrases that do not contain markdown characters like *, #, _ to ensure better matching.
4. Be extremely precise with capitalization and punctuation.

Format: {{"highlights": ["phrase 1", "phrase 2", ...]}}

ANSWER:
"{answer}"

DOCUMENT:
"{document_content}"
"""
        
        # Generate response with JSON mode
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Handle different possible JSON structures
        if isinstance(result, list):
            highlights = result
        elif isinstance(result, dict) and "highlights" in result:
            highlights = result["highlights"]
        elif isinstance(result, dict) and "phrases" in result:
            highlights = result["phrases"]
        else:
            # Try to get any array value from the dict
            for value in result.values():
                if isinstance(value, list):
                    highlights = value
                    break
            else:
                highlights = []
        
        print(f"✓ Highlights generated with Groq: {len(highlights)} phrases")
        return highlights
        
    except Exception as groq_error:
        print(f"⚠ Groq failed, falling back to Gemini: {groq_error}")
        
        # Fallback to Gemini
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                print("✗ GEMINI_API_KEY not set, cannot fallback")
                return []
            
            # Configure Gemini client
            genai.configure(api_key=gemini_api_key)
            
            # Create model with strict system instruction
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                system_instruction="You are a precision text extraction engine. Return ONLY a JSON array of exact verbatim phrases found in the document."
            )
            
            # Craft prompt with strict rules for verbatim extraction
            prompt = f"""Find 5-8 short, key phrases (3-6 words each) in the DOCUMENT that specifically support the claims in the ANSWER.
                
STRICT RULES:
1. Return a JSON array of strings.
2. Each string MUST be a LITERALLY EXACT VERBATIM substring from the DOCUMENT.
3. Choose phrases that do not contain markdown characters like *, #, _ to ensure better matching.
4. Be extremely precise with capitalization and punctuation.

ANSWER:
"{answer}"

DOCUMENT:
"{document_content}"
"""
            
            # Generate response with JSON schema enforcement
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "array",
                        "items": {"type": "string"}
                    }
                )
            )
            
            # Parse and return the array of highlight strings
            highlights = json.loads(response.text)
            print(f"✓ Highlights generated with Gemini (fallback): {len(highlights)} phrases")
            return highlights
            
        except Exception as gemini_error:
            print(f"✗ Gemini fallback failed: {gemini_error}")
            return []
