import openai

API_KEY = open("API_KEY.txt", "r").read()
openai.api_key = API_KEY

response_log = []
user_message = open("../TranscribedText.txt", "r").read()
assistant_response = []

def bot_assistant():
    response_log.append({"role": "user", "content": user_message.strip("\n").strip()})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= response_log
    )
    assistant_response = response['choices'][0]['message']['content']
    print("ChatGPT: ", assistant_response.strip("\n").strip())
    response_log.append({"role": "assistant", "content": assistant_response.strip("\n").strip()})

def main():
    print(user_message)
    bot_assistant()

if __name__ == '__main__':
    main()