import tiktoken


class TokenPricer(object):
    def __init__(self, engine="gpt-3.5-turbo-0301"):
        super(TokenPricer, self).__init__()
        self.gpt_costs_per_thousand = {
            'gpt-3.5-turbo-0301': 0.0015,
            'davinci': 0.0200,
            'curie': 0.0020,
            'babbage': 0.0005,
            'ada': 0.0004
            }

        # Use tiktoken.encoding_for_model() to automatically load the correct encoding for a given model name.
        self.tokenizer = tiktoken.encoding_for_model(engine)
        self.engine = self.engine

    def gpt_get_estimated_cost(self, prompt, max_tokens, config=None):
        """Uses the current API costs/1000 tokens to estimate the cost of generating text from the model."""
        prompt = self.tokenizer.encode(prompt)
        # Get the number of tokens in the prompt
        n_prompt_tokens = len(prompt)
        # Get the number of tokens in the generated text
        total_tokens = n_prompt_tokens + max_tokens
        # engine = config['gpt_config']['model'].split('-')[1]
        costs_per_thousand = self.gpt_costs_per_thousand
        # if engine not in costs_per_thousand:
        #     # Try as if it is a fine-tuned model
        #     engine = config['gpt_config']['model'].split(':')[0]
        #     costs_per_thousand = {
        #         'gpt-3.5-turbo-0301': 0.0015,
        #         'gpt-3.5-turbo-0613': 0.0015,
        #         'davinci': 0.1200,
        #         'curie': 0.0120,
        #         'babbage': 0.0024,
        #         'ada': 0.0016
        #     }
        price = costs_per_thousand[self.engine] * total_tokens / 1000
        return price