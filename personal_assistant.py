import os
import requests
from PIL import Image
from pathlib import Path
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

#tools gemini can call

tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Gets the current weather for a city",
            parameters=types.Schema(
                type="OBJECT",
                properties={"city": types.Schema(type="STRING")},
                required=["city"]
            )
        ),
        types.FunctionDeclaration(
            name="get_joke",
            description="Tells a joke by category (e.g. Programming, Misc)",
            parameters=types.Schema(
                type="OBJECT",
                properties={"category": types.Schema(type="STRING")},
                required=["category"]
            )
        ),
        types.FunctionDeclaration(
            name="convert_units",
            description="Converts a value between units (e.g. miles to km, celsius to fahrenheit)",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "value": types.Schema(type="NUMBER"),
                    "from_unit": types.Schema(type="STRING"),
                    "to_unit": types.Schema(type="STRING")
                },
                required=["value", "from_unit", "to_unit"]
            )
        ),
        types.FunctionDeclaration(
            name="get_fact",
            description="Returns a fun fact about a topic",
            parameters=types.Schema(
                type="OBJECT",
                properties={"topic": types.Schema(type="STRING")},
                required=["topic"]
            )
        ),
        types.FunctionDeclaration(
            name="get_definition",
            description="Returns a simple definition of a word or concept",
            parameters=types.Schema(
                type="OBJECT",
                properties={"word": types.Schema(type="STRING")},
                required=["word"]
            )
        ),
    ])
]

#tool implementations 

def get_weather(city):
    res = client.models.generate_content(
        model=MODEL,
        contents=f"What's the current weather in {city}? Temp in Fahrenheit, short description.",
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
    )
    return {"weather": res.text}

def get_joke(category):
    res = requests.get(f"https://v2.jokeapi.dev/joke/{category}?type=single", timeout=5)
    return {"joke": res.json()["joke"]}

def convert_units(value, from_unit, to_unit):
    conversions = {
        ("miles", "km"): value * 1.60934,
        ("km", "miles"): value / 1.60934,
        ("celsius", "fahrenheit"): value * 9/5 + 32,
        ("fahrenheit", "celsius"): (value - 32) * 5/9,
    }
    result = conversions.get((from_unit.lower(), to_unit.lower()))
    return {"result": round(result, 4)} if result else {"error": "unknown conversion"}

def get_fact(topic):
    res = client.models.generate_content(
        model=MODEL,
        contents=f"Give me one fun fact about {topic}. Just the fact, no extra commentary.",
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
    )
    return {"fact": res.text}

def get_definition(word):
    res = client.models.generate_content(
        model=MODEL,
        contents=f"Give a simple one-sentence definition of: {word}"
    )
    return {"definition": res.text}

tool_map = {
    "get_weather": get_weather,
    "get_joke": get_joke,
    "convert_units": convert_units,
    "get_fact": get_fact,
    "get_definition": get_definition,
}

#ask gemini 

def ask(user_input):
    response = client.models.generate_content(
        model=MODEL,
        contents=user_input,
        config=types.GenerateContentConfig(tools=tools)
    )

    part = response.candidates[0].content.parts[0]

    if part.function_call:
        fn_name = part.function_call.name
        fn_args = dict(part.function_call.args)
        print(f"[tool] {fn_name}({fn_args})")

        result = tool_map[fn_name](**fn_args)

        follow_up = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(role="user", parts=[types.Part(text=user_input)]),
                types.Content(role="model", parts=[part]),
                types.Content(role="tool", parts=[
                    types.Part(function_response=types.FunctionResponse(name=fn_name, response=result))
                ]),
            ],
            config=types.GenerateContentConfig(tools=tools)
        )
        return follow_up.text

    return response.text

#image analysis

def analyze_image(image_path, question):
    img_bytes = Path(image_path).read_bytes()
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_bytes)),
            types.Part(text=question)
        ]
    )
    return response.text

#image generation (opens in VS Cde)

def generate_image(prompt, output_path="output.png"):
    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"]
        )
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data:
            with open(output_path, "wb") as f:
                f.write(part.inline_data.data)
            print(f"image saved as {output_path}")
            os.system(f"code {output_path}")  # opens in VS Code as a tab
            return

    print("no image was returned")

#run it

if __name__ == "__main__":
    print("--- weather ---")
    print(ask("What's the weather in Athens, Ohio?"))

    print("\n--- joke ---")
    print(ask("Tell me a programming joke"))

    print("\n--- convert ---")
    print(ask("Convert 10 miles to km"))

    print("\n--- fact ---")
    print(ask("Give me a fun fact about space"))

    print("\n--- definition ---")
    print(ask("What does 'recursion' mean?"))

    print("\n--- image analysis ---")
    # drop a jpg in the same folder, rename it to test.jpg, or change the path below
    print(analyze_image("test.jpg", "What do you see in this image? Describe it in detail."))

    print("\n--- image generation ---")
    generate_image("a cute robot studying at a desk with coffee and books, digital art")