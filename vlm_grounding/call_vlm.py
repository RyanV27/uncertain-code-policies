import ast
import os
import json
import argparse
import re
import time
import base64

from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, GenerationConfig

# ----------------- Do this if running the script inside test_code/------------------ #
import sys

# Adding parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
# ---------------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Prompts the VLM with the environment images to identify the objects in each scene.")

    parser.add_argument("--provider", type=str, choices=["huggingface", "openai"], default="huggingface", help="Name of the model provider")
    parser.add_argument("--model", type=str, required=True, help="Name/ID of the VLM to test.")
    parser.add_argument(
        "--path",
        type=str,
        default="./runs/vlm_test_envs",
        help=(
            "Path to output directory where environment images and a JSON",
            "file where an object list for each environment is stored."
        )
    )

    args = parser.parse_args()
 
    provider = args.provider
    save_dir = Path(args.path)
    vlm_id = args.model
    model_save_dir = save_dir / vlm_id if provider == "huggingface" else save_dir / provider / vlm_id

    # Creating a folder to store the particular model's results
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_results_file_name = "vlm_env_obj_lists.json"
    model_results_file_path = model_save_dir / model_results_file_name
     
    if provider == "huggingface":
        # Loading the model, processor and generation config
        object_id_processor = AutoProcessor.from_pretrained(vlm_id, trust_remote_code=True, cache_dir="/scratch/rsvargh2/huggingface_models/")
        object_id_model = AutoModelForCausalLM.from_pretrained(
            vlm_id, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            _attn_implementation='eager',
            cache_dir="/scratch/rsvargh2/huggingface_models/"
        ).cuda()
        
        object_id_gen_config = GenerationConfig.from_pretrained(vlm_id, cache_dir="/scratch/rsvargh2/huggingface_models/")
    else:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Loading OpenAI APU client
        client = OpenAI(api_key=openai_api_key)
        

    object_id_prompt = f"<|user|><|image_1|>Environment: The environment consists of a robotic arm and a black surface. The black surface may have bowls or blocks of different colors on top of it. Instructions: 1. From the given image, identify all the bowls and blocks along with their colors. 2. Return a Python list of strings as your output. Each string should have the format 'color object_type'. 3. Think step by step. However, only return the required Python list of strings. Options for color: blue, red, green, orange, yellow, purple, pink, cyan, brown, gray. Options for object_type: block, bowl.<|end|><|assistant|>"

    model_results = []

    # Listing all the json files in sorted order
    files = [filename for filename in os.listdir(save_dir) if filename.endswith(".json")]
    files = sorted(files, key=lambda f: int(re.search(r'env_(\d+)_obj_list', f).group(1)))
    
    for filename in files:
        with open(save_dir / filename, "r") as f_in:
            cur_env_data = json.load(f_in)

        print(f"Env {cur_env_data['id']}")
        print(f"Available objects: {cur_env_data['actual']}")

        # Reading the corresponding environment image
        try:
            if provider == "huggingface":
                env_img = Image.open(save_dir / f"env_{cur_env_data['id']}_img.jpg")       # Pillow image for Hugging Face models
            elif provider == "openai":
                with open(save_dir / f"env_{cur_env_data['id']}_img.jpg", "rb") as image_file:     # Base64-encoded string for OpenAI models
                    b64_env_img = base64.b64encode(image_file.read()).decode("utf-8")
                    
            cur_env_data['image_read'] = True
        except Exception as e:
            print("Failed to read image!")
            print(f"Error: {e}")
            cur_env_data['image_read'] = False
            cur_env_data['predicted'] = []
            cur_env_data['correct_format'] = False
            continue

        # Generate response from the VLM
        if provider == "huggingface":
            inputs = object_id_processor(text=object_id_prompt, images=env_img, return_tensors='pt').to('cuda:0')
    
            generate_ids = object_id_model.generate(
                **inputs,
                max_new_tokens=1000,
                generation_config=object_id_gen_config,
            )
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = object_id_processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            print(f"Response: {response}")
        elif provider == "openai":
            # Due to a limit on the number of calls per minute, have to retry after a time interval.
            while True:
                try:
                    response = client.chat.completions.create(
                        model=vlm_id,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": object_id_prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpg;base64,{b64_env_img}"
                                        }
                                    }
                                ]
                            }
                        ]
                    ).choices[0].message.content
                    print(f"Response: {response}")
                    break
                except Exception as e:
                    print("Could not get response from GPT. Trying again after 20 seconds.")
                    print(f"Error: {e}")
                    time.sleep(20)

        # Checking if the output is Python parsable
        try:
            pred_obj_list = ast.literal_eval(response)
            cur_env_data['predicted'] = pred_obj_list
            cur_env_data['correct_format'] = True
        except:
            print("Incorrect format from the VLM.")
            cur_env_data['predicted'] = []
            cur_env_data['correct_format'] = False
        
        print(f"Predicted objects: {cur_env_data['predicted']}\n")

        model_results.append(cur_env_data)

    # Saving the VLM's results
    with open(model_results_file_path, "w") as f_out:
        json.dump(model_results, f_out)

    print(f"Saved the results in {model_results_file_path}.")

if __name__ == "__main__":
    main()