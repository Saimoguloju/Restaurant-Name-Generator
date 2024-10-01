from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

llm = ChatGroq(
    model_name="llama-3.2-90b-text-preview",  
    temperature=0,
    groq_api_key="API-KEY"
)

def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )

    # Create the first LLM chain for generating a restaurant name
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    # Define the prompt template for suggesting menu items based on the restaurant name
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}."
    )

    # Create the second LLM chain for generating menu items
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")
    
    chain = SequentialChain(
        chains = [name_chain,food_items_chain],
        input_variables = ['cuisine'],
        output_variables = ['restaurant_name','menu_items']
    )
       
       
    response =   chain({'cuisine' : cuisine})
    
    return response

if __name__ == "__main__":
    print(generate_restaurant_name_and_items('Italian'))
