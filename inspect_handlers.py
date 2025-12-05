import llama_cpp.llama_chat_format
import inspect

print("Available handlers in llama_cpp.llama_chat_format:")
for name, obj in inspect.getmembers(llama_cpp.llama_chat_format):
    if inspect.isclass(obj) and "Handler" in name:
        print(name)
