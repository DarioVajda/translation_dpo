import torch
print("imported torch version:", torch.__version__)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
print("imported transformers")

# Path to your local Hugging Face 9B parameter model
MODEL_PATH = "/ceph/hpc/data/s24o01-42-users/models/hf_models/GaMS-9B-Instruct-translate-v1"
# MODEL_PATH = "/ceph/hpc/data/s24o01-42-users/models/hf_models/GaMS-9B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Tokenizer loaded successfully.")

# Load model with automatic device mapping (will spread across GPUs if needed)
# Use half-precision for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    # device_map=0,           # Automatically place layers on available GPUs
    torch_dtype=torch.float16,     # Use fp16
    low_cpu_mem_usage=True         # Reduce CPU memory usage during init
)
print("cuda" if torch.cuda.is_available() else "cpu")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available
print("Model loaded successfully.")

# Create a text-generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map=0           # Ensure pipeline uses the same device mapping
)


def generate_response(prompt: str, max_new_tokens: int = 4096, **gen_kwargs) -> str:
    """
    Generates a response given an input prompt.

    Args:
        prompt (str): The input text prompt.
        max_new_tokens (int): Maximum tokens to generate.
        **gen_kwargs: Additional generation parameters.

    Returns:
        str: The generated text.
    """
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=gen_kwargs.get("do_sample", True),
        temperature=gen_kwargs.get("temperature", 0.8),
        top_p=gen_kwargs.get("top_p", 0.95),
        top_k=gen_kwargs.get("top_k", 50),
    )
    return output[0]["generated_text"]


if __name__ == "__main__":
    # Input text prompt
    user_prompt = (
        "Prevedi naslednje angleško besedilo v slovenščino.\n" +
        "# Vallcarca metro station\n" +
        "\n" +
        "Vallcarca is a Barcelona Metro station in the Vallcarca i els Penitents neighbourhood, in the Gràcia district of Barcelona.The station is served by line L3.\n" +
        "\n" +
        "The station opened in 1985 when the section of line L3 between Lesseps and Montbau stations was inaugurated.\n" +
        # "\n" +
        # "The station is located underneath Avinguda de Vallcarca (formerly known as the Avinguda de l'Hospital Militar), between Carrer de l'Argentera and the Vallcarca bridge. It has three entrances and can be accessed from either side of Avinguda de Vallcarca, as well as from Avinguda de la República Argentina. It has twin side platforms that are long and which are accessed from the entrance lobby by stairs and escalators.\n" +
        # "\n" +
        # "## See also\n" +
        # "\n" +
        # "* List of Barcelona Metro stations\n" +
        # "\n" +
        # "## External links\n" +
        # "\n" +
        # "* \n" +
        # "\n" +
        # "* Trenscat.com\n" +
        # "\n" +
        # "* Transportebcn.es\n" +
        # "\n" +
        # "Barcelona Metro line 3 stations\n" +
        # "Railway stations in Spain opened in 1985\n" +
        # "Transport in Gràcia\n" + 
        "Prevod: " +
        ""
    )

    responses = []
    for i in range(2):
        # Generate response
        response = generate_response(
            user_prompt,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9
        )
        responses.append(response)

    # Original prompt display
    print("\n=== User Prompt ===")
    print(user_prompt)
    for response in responses:
        # Print model output
        print("\n=== Model Response ===")
        print(response)

# {
#   "rejected_score":0.7719166279,
#   "rejected":"# Postaja metroja Vallcarca\n\nVallcarca je postaja metroja v Barceloni, ki služi soseski Vallcarca i els Penitents, v okrožju Gràcia v Barceloni. Postajo upravlja linija L3.\n\nPostaja je bila odprta leta 1985, ko je bila odprta linija L3 med postajama Lesseps in Montbau.\n\nPostaja se nahaja pod Avinguda de Vallcarca (nekdanja Avinguda de l'Hospital Militar), med Carrer de l'Argentera in mostom Vallcarca. Ima tri vhode, dostopne z obeh strani Avinguda de Vallcarca ter preko Avinguda de la República Argentina. Ima dve bočni peroni, dolgi 75 metrov, dostopni preko stopnic in dvigal.\n\n## Glej tudi\n\n* Seznam postaj metroja Barcelona\n\n## Zunanje povezave\n\n* \n\n* Trenscat.com\n\n* Transportebcn.es\n",
#   "prompt":"<bos><start_of_turn>user\nPrevedi naslednje angleško besedilo v slovenščino.\n# Vallcarca metro station\n\nVallcarca is a Barcelona Metro station in the Vallcarca i els Penitents neighbourhood, in the Gràcia district of Barcelona.The station is served by line L3.\n\nThe station opened in 1985 when the section of line L3 between Lesseps and Montbau stations was inaugurated.\n\nThe station is located underneath Avinguda de Vallcarca (formerly known as the Avinguda de l'Hospital Militar), between Carrer de l'Argentera and the Vallcarca bridge. It has three entrances and can be accessed from either side of Avinguda de Vallcarca, as well as from Avinguda de la República Argentina. It has twin side platforms that are long and which are accessed from the entrance lobby by stairs and escalators.\n\n## See also\n\n* List of Barcelona Metro stations\n\n## External links\n\n* \n\n* Trenscat.com\n\n* Transportebcn.es\n\nBarcelona Metro line 3 stations\nRailway stations in Spain opened in 1985\nTransport in Gràcia<end_of_turn>\n<start_of_turn>model\n",
#   "chosen_score":0.8264608383,
#   "chosen":"# Metro postaja Vallcarca\n\nVallcarca je postaja Barcelonskega metroja v soseski Vallcarca i els Penitents v okrožju Gràcia v Barceloni. Postajo obratuje linija L3.\n\nPostaja je bila odprta leta 1985, ko je bil odprt odsek linije L3 med postajama Lesseps in Montbau.\n\nPostaja se nahaja pod Avenido de Vallcarca (prej imenovana Avenida de l'Hospital Militar), med ulico Carrer de l'Argentera in mostom Vallcarca. Ima tri izhode in dostop z obeh strani Avenide de Vallcarca, pa tudi z Avenide de la República Argentina. Sestavljena je iz dveh stranskih peronov, ki sta dolga in dostopna iz vhodne avle po stopnicah in dvigalih.\n\n## Glej tudi\n\n* Seznam postaj Barcelonskega metroja\n\n## Zunanje povezave\n\n* \n\n* Trenscat.com\n\n* Transportebcn.es\n\nPostaje Barcelonskega metroja na liniji 3\nŽelezniške postaje v Španiji, odprte leta 1985\nTransport v okrožju Gràcia",
#   "src":"eurollm"
# }