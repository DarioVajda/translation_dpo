from transformers import pipeline

# model_id = "cjvt/GaMS-9B-Instruct"
model_id = "/ceph/hpc/data/s24o01-42-users/models/hf_models/GaMS-9B-Instruct-translate-v1"

pline = pipeline(
    "text-generation",
    model=model_id,
    device_map="cuda" # replace with "mps" to run on a Mac device
)

# Example of response generation
message = [{
    "role": "user",
    "content": 
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
}]
response = pipeline(message, max_new_tokens=2048)
print("Model's response:", response[0]["generated_text"][-1]["content"])

# Example of conversation chain
# new_message = response[0]["generated_text"]
# new_message.append({"role": "user", "content": "Lahko bolj podrobno opišeš ta dogodek?"})
# response = pipeline(new_message, max_new_tokens=2048)
# print("Model's response:", response[0]["generated_text"][-1]["content"])