 stream = client.chat.completions.create(
            model=MODEL,
            messages=st.session_state.messages,
            stream=True,
        )
