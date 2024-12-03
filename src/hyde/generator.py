import time
import ollama

class Generator:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
    
    def generate(self):
        return ""

class OllamaGenerator(Generator):
    def __init__(self, model_name, temperature=0.7):
        super().__init__(model_name, None)  
        self.temperature=temperature

    @staticmethod
    def parse_response(response):
        return response.get('response')

    def generate(self, prompt, n=8):
        texts = []
        for _ in range(n):
            get_result = False
            while not get_result:
                try:
                    # Assuming the correct method is something like `complete`
                    result = ollama.generate(  # Use the correct method from the Ollama library
                        model=self.model_name,
                        prompt=prompt,
                        options={"temperature": self.temperature}
                    )
                    get_result = True
                except Exception as e:
                    if self.wait_till_success:
                        time.sleep(1)
                    else:
                        raise e
            text = self.parse_response(result)
            texts.append(text)
        return texts
    
    def get_temperature(self):
        return self.temperature