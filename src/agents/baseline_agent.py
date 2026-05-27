import ollama


class BaselineAgent:
    def __init__(
        self,
        model_name: str = "mistral:7b-instruct",
    ) -> None:
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]
