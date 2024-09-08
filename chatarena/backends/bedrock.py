import os
import re
import json
from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential
from ..message import SYSTEM_NAME, Message
from .base import IntelligenceBackend, register_backend
import boto3
from botocore.config import Config
from . import bedrock_guardrail

DEFAULT_REGION = 'us-west-2'
is_bedrock_available = False

try:
    # Check if the necessary AWS credentials are set
    session = boto3.Session()
    bedrock = session.client(
        service_name='bedrock-runtime',
        region_name=os.environ.get('AWS_DEFAULT_REGION', DEFAULT_REGION),
        config=Config(retries={'max_attempts': 3, 'mode': 'standard'})
    )
    bedrockRuntimeClient = boto3.client('bedrock-runtime', region_name=DEFAULT_REGION)

    is_bedrock_available = True
except Exception:
    is_bedrock_available = False

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
END_OF_MESSAGE = ""
STOP = ("<|endoftext|>", END_OF_MESSAGE)
BASE_PROMPT = f"The messages always end with the token {END_OF_MESSAGE}."

guardrailid, guardrailversion = bedrock_guardrail.setup_guardrail()



@register_backend
class BedrockClaude(IntelligenceBackend):
    """Interface to the Bedrock Claude 3 model with system, user, assistant roles separation."""

    stateful = False
    type_name = "bedrock-claude"

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: str = DEFAULT_MODEL,
        system_prompt: str="",

        merge_other_agents_as_one_user: bool = True,

        **kwargs,
    ):
        assert is_bedrock_available, "boto3 package is not installed or Bedrock is not configured"
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            merge_other_agents_as_one_user=merge_other_agents_as_one_user,
            **kwargs,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.merge_other_agent_as_user = merge_other_agents_as_one_user
        self.guardrail_id: str = guardrailid
        self.guardrail_version: str = guardrailversion
    

    def apply_guardrails(self, input_text: str) -> tuple:
        if not self.guardrail_id or not self.guardrail_version:
            return input_text, "ALLOW"
        # content_str = [{"text": {input_text}}]
        input = input_text
        content_str=[{"text": {"text": input}}]
        try:
            response = bedrockRuntimeClient.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source='INPUT',
                content=content_str
            )
            
            print(f'Guradrail respoinse: {response}')
            guardrailResult = response["action"]
            return guardrailResult
        
            
        except Exception as e:
            print(f"Error applying guardrail: {e}")
            return input_text, "ALLOW"


    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": messages,
            "temperature": self.temperature,
        }
        print(f"--------------------")
        print(f"prompt is {messages}")
        print(f"--------------------")
        response = bedrock.invoke_model(
            body=json.dumps(body),
            modelId=self.model,
            contentType="application/json",
            accept="application/json",
        )
        
        # response_body = response['body'].read()
        print(f"------response_str START-------")
        response_body = json.loads(response['body'].read())
        response_text = response_body['content'][0]['text']
        print(f"response text is ------> {response_text}")
        return response_text

    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:
        if global_prompt:
            self.system_prompt = f"You are a player in the game, please do not reveal your identity.\n{global_prompt.strip()}\n{BASE_PROMPT}\n\nYour name is {agent_name}.\n\nYour role:{role_desc}"
        else:
            self.system_prompt = f"You are a player in the game, please do not reveal your identity. Your name is {agent_name}.\n\nYour role:{role_desc}\n\n{BASE_PROMPT}"
    
        messages = []
    
        last_role = "system"
        for msg in history_messages:
                
            if msg.agent_name == SYSTEM_NAME:
                if last_role == "user":
                    messages.append({"role": "assistant", "content": "Understood."})
                messages.append({"role": "user", "content": msg.content})
                last_role = "user"
            else:
                role = "assistant" if msg.agent_name == agent_name else "user"
                 ## --- 插入guardrail check -------
                _guardrail_action = self.apply_guardrails(msg.content)
                if _guardrail_action != "NONE":
                    print(f"_guardrail_action block {msg.content}")
                    msg.content ="该发言内容敏感，无法显示"
                if role == last_role:
                    messages[-1]["content"] += f"\n\n[{msg.agent_name}]: {msg.content}{END_OF_MESSAGE}"
                else:
                    messages.append({"role": role, "content": f"[{msg.agent_name}]: {msg.content}{END_OF_MESSAGE}"})
                last_role = role

    
        if request_msg:
            if last_role == "user":
                messages.append({"role": "assistant", "content": "Understood."})
            messages.append({"role": "user", "content": request_msg.content})
        else:
            if last_role == "user":
                messages.append({"role": "assistant", "content": "Understood."})
            request_msg_agent = {"role": "user", "content": f"Now you speak, {agent_name}.{END_OF_MESSAGE}"}
            messages.append(request_msg_agent)

        response = self._get_response(messages, *args, **kwargs)
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()
        response = re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", response).strip()
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()
        return response