from typing import Optional, List, Mapping, Any
import requests
from time import sleep
from .base import LLM
from .pyChatGPT import ChatGPT as Chatbot
from typing import Optional, List, Mapping, Any
import requests
from time import sleep
from .base import LLM
import ast

class ChatGPT(LLM):
    """ChatGPT LLM"""

    history_data: Optional[List] = []
    api_token: Optional[str]
    chatbot: Optional[Chatbot] = None
    call_count: int = 0
    conversation: Optional[str] = ""

    def __init__(self, api_token: Optional[str] = None, conversation_id: Optional[str] = None):
        if api_token is not None:
            self.api_token = api_token
        if conversation_id is not None:
            self.conversation = conversation_id

    def query(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        if self.chatbot is None:
            if self.api_token is None:
                raise ValueError("Need a token, check https://chat.openai.com/api/auth/session for get your token")
            else:
                if self.conversation == "":
                    self.chatbot = Chatbot(self.api_token)
                elif self.conversation != "":
                    self.chatbot = Chatbot(self.api_token, conversation_id=self.conversation)
                else:
                    raise ValueError("Something went wrong")

        response = ""
        if self.call_count >= 45:
            raise ValueError("You have reached the maximum number of requests per hour! Help me to Improve. Abusing this tool is at your own risk")
        else:
            sleep(2)
            data = self.chatbot.send_message(prompt)
            response = data["message"]
            self.conversation = data["conversation_id"]
            FullResponse = data

            self.call_count += 1

        self.history_data.append({"prompt": prompt, "response": response})

        return response

    def call(self, instruction: str, value: str) -> str:
        output = self.query(instruction + value)
        code = self.extract_python_code(output)
        return code
    

    def extract_python_code(self, multiline_string):
        python_code_lines = []
        
        for line in multiline_string.split('\n'):
            line = line.strip()
            try:
                ast.parse(line)
                python_code_lines.append(line)
            except SyntaxError:
                continue
        
        python_code = "\n".join(python_code_lines)
        return python_code
    


    @property
    def type(self) -> str:
        return "custom"