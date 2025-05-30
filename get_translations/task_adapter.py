import warnings

class TaskAdapter:
    def __init__(self):
        pass

    def create_prompt(self, input):
        pass


class GaMSTaskAdapter:
    def __init__(self):
        self.instruction = "Prevedi naslednje angleško besedilo v slovenščino."

    def create_prompt(self, input):
        user_message = f"{self.instruction}\n{input}"
        conversation = [
            {"role": "user", "content": user_message}
        ]

        return conversation
    

class EuroLLMTaskAdapter:
    def __init__(self):
        self.instruction = "Translate the following English text to Slovenian."

    def create_prompt(self, input):
        conversation = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": input}
        ]

        return conversation


def get_task_adapter(model_name) -> TaskAdapter:
    if "GaMS" in model_name:
        return GaMSTaskAdapter()
    elif "EuroLLM" in model_name:
        return EuroLLMTaskAdapter()
    
    raise ValueError("Unsupported model name", model_name)
