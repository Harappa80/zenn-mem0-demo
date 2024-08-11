from mem0_chat import Mem0Chat

if __name__ == "__main__":
    user_id = input("Prease input your user_id:")
    mem0_chatbot = Mem0Chat(
        user_id=user_id,
        app_id="mem0_chat",
    )

    # 初回の会話(LLMから会話を開始するために定義)
    mem0_chatbot.gen_response("こんにちは", user_id=mem0_chatbot.user_id)

    # マルチターンチャット
    while True:
        try:
            user_input = input(">>")
            mem0_chatbot.gen_response(
                user_input,
                user_id=mem0_chatbot.user_id,
            )
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
            break

        # 現在保存されている記憶を表示
        memories = mem0_chatbot.get_memories(user_id=mem0_chatbot.user_id)
        for m in memories:
            print(m["memory"])
