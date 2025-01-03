import logging
import warnings
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
import litellm

@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            yield
        finally:
            pass

class MistralCrewLLM:
    def __init__(
        self,
        model: str,
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        apikey: Optional[str] = None,  # Changed from api_key to match your init
        callbacks: List[Any] = [],
        **kwargs,
    ):
        self.model = f"mistral/{model}"  # Prefix with mistral/
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop = stop
        self.max_completion_tokens = max_completion_tokens
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.response_format = response_format
        self.seed = seed
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.base_url = base_url
        self.api_version = api_version
        self.api_key = apikey
        self.callbacks = callbacks
        self.kwargs = kwargs
        
        litellm.drop_params = True
        self.set_callbacks(callbacks)

    def call(self, messages: List[Dict[str, str]], callbacks: List[Any] = []) -> str:
        with suppress_warnings():
            if callbacks and len(callbacks) > 0:
                self.set_callbacks(callbacks)

            try:
                params = {
                    "model": self.model,
                    "messages": messages,
                    "timeout": self.timeout,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "n": self.n,
                    "stop": self.stop,
                    "max_tokens": self.max_tokens or self.max_completion_tokens,
                    "presence_penalty": self.presence_penalty,
                    "frequency_penalty": self.frequency_penalty,
                    "logit_bias": self.logit_bias,
                    "response_format": self.response_format,
                    "seed": self.seed,
                    "logprobs": self.logprobs,
                    "top_logprobs": self.top_logprobs,
                    "api_base": self.base_url,
                    "api_version": self.api_version,
                    "api_key": self.api_key,
                    "stream": False,
                    **self.kwargs,
                }

                # Remove None values
                params = {k: v for k, v in params.items() if v is not None}

                response = litellm.completion(**params)
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                logging.error(f"LiteLLM call failed: {str(e)}")
                raise

    def set_callbacks(self, callbacks: List[Any]):
        callback_types = [type(callback) for callback in callbacks]
        for callback in litellm.success_callback[:]:
            if type(callback) in callback_types:
                litellm.success_callback.remove(callback)

        for callback in litellm._async_success_callback[:]:
            if type(callback) in callback_types:
                litellm._async_success_callback.remove(callback)

        litellm.callbacks = callbacks

    def supports_function_calling(self) -> bool:
        try:
            params = litellm.get_supported_openai_params(model=self.model)
            return "response_format" in params
        except Exception as e:
            logging.error(f"Failed to get supported params: {str(e)}")
            return False

    def supports_stop_words(self) -> bool:
        try:
            params = litellm.get_supported_openai_params(model=self.model)
            return "stop" in params
        except Exception as e:
            logging.error(f"Failed to get supported params: {str(e)}")
            return False

    def get_context_window_size(self) -> int:
        return int(128000 * 0.75)  # Using 75% of Mistral's context window

    def __repr__(self):
        return f"MistralCrewLLM(model={self.model}, temperature={self.temperature})"