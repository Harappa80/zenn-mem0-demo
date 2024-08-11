import os
import textwrap
from os import getenv

import google.generativeai as genai
from dotenv import load_dotenv
from mem0 import Memory

# 環境変数の設定
load_dotenv()
os.environ["OPENAI_API_KEY"] = getenv("OPENAI_API_KEY", "")
genai.configure(api_key=getenv("GEMINI_API_KEY", ""))


class Mem0Chat:
    def __init__(self, user_id=None, app_id=None):
        self.app_id = app_id
        self.user_id = user_id

        # mem0の初期化（要約で使用するLLM、VectorStore、embeddingモデルを設定）
        mem0_config = {
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": "gemini/gemini-1.5-pro-latest",
                    "temperature": 0.1,
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "test",
                    "path": "db",
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                },
            },
        }
        self.memory = Memory.from_config(mem0_config)

        # 過去の記憶を取得して、システムプロンプトを生成
        memories = self.get_memories(user_id=self.user_id)
        if memories:
            history = self.__convert_memories_to_str(memories)
            system_prompt = (
                textwrap.dedent(
                    """
                    あなたはカスタマーサポートエージェントです。ユーザーの商品購入のサポートをします。
                    下記は、過去の会話の要約です。これを踏まえてユーザーのサポートしてください。
                    {history}
                    """
                )
                .format(history=history)
                .strip()
            )
        else:
            system_prompt = textwrap.dedent(
                """\
                あなたはカスタマーサポートエージェントです。ユーザーの商品購入のサポートをします。
                """
            )
        print(system_prompt)

        # チャットに使用するLLMの初期化
        chat_llm_config = {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-exp-0801",
            generation_config=chat_llm_config,
            system_instruction=system_prompt,
        )
        self.chat_session = self.model.start_chat(history=[])

    # ユーザーの入力を処理
    def gen_response(self, query, user_id=None):
        response = self.chat_session.send_message(query)
        print(response.text)

        self.memory.add(
            query,
            user_id=user_id,
            metadata={"app_id": self.app_id},
        )

    # mem0から会話履歴を全件取得
    def get_memories(self, user_id=None):
        return self.memory.get_all(
            user_id=user_id,
        )

    # mem0の会話履歴を検索取得
    def search_memories(self, query=None, user_id=None):
        return self.memory.search(
            query=query,
            user_id=user_id,
        )

    # 取得した記憶情報を整形
    def __convert_memories_to_str(self, memories):
        memory_values = [
            f"- {m["memory"].replace("\\n", "\n")}" for m in memories
        ]
        return "\n".join(memory_values)
