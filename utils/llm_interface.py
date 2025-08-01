"""
LLM Interface for Cerebras API Communication.
Handles asynchronous communication with the Cerebras LLM.
"""

import json
from typing import Dict, Any
from cerebras.cloud.sdk import AsyncCerebras
from config.settings import Settings

class LLMInterface:
    """
    Interface for communicating with Cerebras LLM API.
    """

    settings = Settings()

    def __init__(self):
        """
        Initialize the LLM interface.
        
        Args:
            api_key: Cerebras API key
            model_name: Name of the Cerebras model
        """
        self.api_key = self.settings.CEREBRAS_API_KEY
        self.model_name = self.settings.LLM_MODEL_NAME
        self.client = AsyncCerebras(api_key=self.api_key)
        
    async def query_llm(self, prompt: str, response_schema: Dict[str, Any] = None) -> Dict[str, Any]:
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
                # Optional: Validate against response_schema if provided
                if response_schema:
                    for key in response_schema:
                        if key not in parsed_content:
                            raise ValueError(f"Missing expected key {key} in LLM response")
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
                        if response_schema:
                            for key in response_schema:
                                if key not in parsed_content:
                                    raise ValueError(f"Missing expected key {key} in LLM response")
                        return parsed_content
                    except:
                        pass
                raise ValueError("LLM returned invalid JSON")
                
        except Exception as e:
            print(f"Error querying LLM: {e}")
            raise
