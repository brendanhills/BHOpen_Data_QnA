# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Provides the base class for all Agents 
"""

from abc import ABC
import vertexai
from google.cloud.aiplatform import telemetry
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import CodeGenerationModel
from vertexai.language_models import CodeChatModel
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import HarmCategory,HarmBlockThreshold
from vertexai.generative_models import Part, SafetySetting, FinishReason


import logging
logger = logging.getLogger(__name__)

from utilities import PROJECT_ID, PG_REGION
vertexai.init(project=PROJECT_ID, location=PG_REGION)



class Agent(ABC):
    """
    The core class for all Agents
    """

    agentType: str = "Agent"

    def __init__(self,
                model_id:str):
        """
        Args:
            PROJECT_ID (str | None): GCP Project Id.
            dataset_name (str): 
            TODO
        """

        self.model_id = model_id 
        logger.info(f"LLM Model is {model_id} ")
        if model_id == 'code-bison-32k':
            self.model = CodeGenerationModel.from_pretrained('code-bison-32k')
        elif model_id == 'text-bison-32k':
            self.model = TextGenerationModel.from_pretrained('text-bison-32k')
        elif model_id == 'codechat-bison-32k':
            self.model = CodeChatModel.from_pretrained("codechat-bison-32k")
        elif model_id.startswith('gemini'):
            with telemetry.tool_context_manager('opendataqna'):
                self.model_config = {
                    "max_output_tokens": 8192,
                    "temperature": 1.0,
                    "top_p": 0.95
                }

                self.safety_settings = [
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    ),
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    ),
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    ),
                    SafetySetting(
                        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    )
                ]

                
                self.model = GenerativeModel(model_id, generation_config=self.model_config)

        else:
            raise ValueError(f"Please specify a compatible model.  Passed Model ID: {model_id}")

    def generate_llm_response(self,prompt):
        context_query = self.model.generate_content(prompt,stream=False)
        return str(context_query.candidates[0].text).replace("```sql", "").replace("```", "")