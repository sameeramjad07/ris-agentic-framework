"""
LLM Interface for Cerebras API Communication.
Handles asynchronous communication with the Cerebras LLM.
"""

import json
from typing import Dict, Any
from cerebras.cloud.sdk import AsyncCerebras

class LLMInterface:
    """
    Interface for communicating with Cerebras LLM API.
    """
    
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize the LLM interface.
        
        Args:
            api_key: Cerebras API key
            model_name: Name of the Cerebras model
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = AsyncCerebras(api_key=api_key)
        
    async def query_llm(self, prompt: str, response_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the Cerebras LLM with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            response_schema: Expected response schema (for validation)
            
        Returns:
            Parsed JSON response from the LLM
        """
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in RIS communication systems. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=0.2,
                max_tokens=1000
            )
            
            # Extract the content from the response
            content = chat_completion.choices[0].message.content
            
            # Parse the JSON content
            try:
                parsed_content = json.loads(content)
                return parsed_content
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM JSON response: {e}")
                print(f"Raw content: {content}")
                # Try to extract JSON from the content if it contains other text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        parsed_content = json.loads(json_match.group())
                        return parsed_content
                    except:
                        pass
                raise ValueError("LLM returned invalid JSON")
                
        except Exception as e:
            print(f"Error querying LLM: {e}")
            raise
