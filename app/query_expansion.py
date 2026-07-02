"""
Conversation Intent Reconstructor

A lossy compression layer that reconstructs retrieval-relevant business intent 
from the complete conversation history while removing conversational filler and 
preserving active constraints.

It is NOT responsible for:
- product recommendation
- assessment selection
- ranking
- retrieval
- grounding

Its sole responsibility is producing a retrieval-oriented representation 
of the recruiter's intent.
"""

import os
import json
import logging
import asyncio
import time
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Conversation Intent Reconstructor for an assessment retrieval system.
Your job is to act as a lossy compression layer. Reconstruct the retrieval-relevant business intent from the complete conversation history, removing conversational filler while preserving all active constraints.

RULES:
1. Output ONLY a valid JSON object matching the schema below. No explanations.
2. The `conversation_summary` must preserve important domain-specific nouns, constraints, and requirements (e.g. "Sales organization undergoing restructuring. Annual talent audit. Maintain the current shortlist while refining for leadership potential and reskilling."). Do NOT write abstract summaries like "The recruiter wants an appropriate assessment."
3. Preserve all active constraints in the `constraints` array (e.g., "Bilingual", "Remote", "Senior", "Technical").
4. Never recommend assessments, infer product categories, or inject retrieval-specific concepts. Only reconstruct what the recruiter is trying to accomplish.
5. If the user explicitly rejects an idea, ensure the summary reflects that constraint (e.g. "without personality tests").

OUTPUT SCHEMA:
{
  "role": "string (the role being hired for, e.g. Sales Manager, or empty if unknown)",
  "objective": "string (the business objective, e.g. Talent audit, Graduate hiring, or empty if unknown)",
  "constraints": ["string", "string"],
  "conversation_action": "string (e.g. clarify, recommend, refine, compare, confirm, reject)",
  "conversation_summary": "string (concise summary of accumulated recruiter intent)"
}
"""

async def reconstruct_intent_async(messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """
    Calls the LLM to reconstruct the business intent from the conversation history.
    Returns the parsed JSON dict on success, or None on failure/fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
        
    start_time = time.time()
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Convert the message history into a readable transcript for the LLM
        transcript = ""
        for m in messages:
            role_label = "Recruiter" if m["role"] == "user" else "Assistant"
            transcript += f"{role_label}: {m['content']}\n\n"
            
        chat_completion = await asyncio.wait_for(
            client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"CONVERSATION TRANSCRIPT:\n{transcript}"}
                ],
                model="gpt-4.1-nano",
                temperature=0.0,
                max_tokens=250,
                response_format={"type": "json_object"}
            ),
            timeout=8.0
        )
        
        response_text = chat_completion.choices[0].message.content
        parsed = json.loads(response_text)
        
        latency = time.time() - start_time
        logger.info("Intent Reconstruction Complete (%.2fs):\n%s", latency, json.dumps(parsed, indent=2))
        
        return parsed
        
    except Exception as e:
        latency = time.time() - start_time
        logger.error("Intent Reconstruction failed (%.2fs): %s", latency, e)
        return None
