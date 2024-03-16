import os
import sys


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)


try:
    import pygit2
    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

    repo = pygit2.Repository(os.path.abspath(os.path.dirname(__file__)))

    branch_name = repo.head.shorthand

    remote_name = 'origin'
    remote = repo.remotes[remote_name]

    remote.fetch()

    local_branch_ref = f'refs/heads/{branch_name}'
    local_branch = repo.lookup_reference(local_branch_ref)

    remote_reference = f'refs/remotes/{remote_name}/{branch_name}'
    remote_commit = repo.revparse_single(remote_reference)

    merge_result, _ = repo.merge_analysis(remote_commit.id)

    if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
        print("Already up-to-date")
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
        local_branch.set_target(remote_commit.id)
        repo.head.set_target(remote_commit.id)
        repo.checkout_tree(repo.get(remote_commit.id))
        repo.reset(local_branch.target, pygit2.GIT_RESET_HARD)
        print("Fast-forward merge")
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
        print("Update failed - Did you modify any file?")
except Exception as e:
    print('Update failed.')
    print(str(e))

import os
import sys
import ssl

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context


import platform
import fooocus_version

from build_launcher import build_launcher
from modules.launch_util import is_installed, run, python, run_pip, requirements_met
from modules.model_loader import load_file_from_url


REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if TRY_INSTALL_XFORMERS:
        if REINSTALL_ALL or not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.23')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    return


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v3.1.safetensors',
     'https://huggingface.co/lllyasviel/misc/resolve/main/xl-to-v1_interposer-v3.1.safetensors')
]

def ini_args():
    from args_manager import args
    return args


prepare_environment()
build_launcher()
args = ini_args()


if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)


from modules import config

def download_models():
    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

    if args.disable_preset_download:
        print('Skipped model download.')
        return

    if not args.always_download_new_model:
        if not os.path.exists(os.path.join(config.paths_checkpoints[0], config.default_base_model_name)):
            for alternative_model_name in config.previous_default_models:
                if os.path.exists(os.path.join(config.paths_checkpoints[0], alternative_model_name)):
                    print(f'You do not have [{config.default_base_model_name}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    config.checkpoint_downloads = {}
                    config.default_base_model_name = alternative_model_name
                    break

    for file_name, url in config.checkpoint_downloads.items():
        load_file_from_url(url=url, model_dir=config.paths_checkpoints[0], file_name=file_name)
    for file_name, url in config.embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in config.lora_downloads.items():
        load_file_from_url(url=url, model_dir=config.paths_loras[0], file_name=file_name)

    return


download_models()

import gradio as gr
import modules.gradio_hijack as grh
from extras.interrogate import default_interrogator as default_interrogator_photo
from extras.wd14tagger import default_interrogator as default_interrogator_anime
import modules.flags as flags

def interrogatorFunction(img, value):
    if value == flags.desc_type_photo:  # Menggunakan operator perbandingan '==' untuk memeriksa kesamaan
        output = default_interrogator_photo(img)
        print(output)
    else:
        output = default_interrogator_anime(img)
        print(output)
    return output

describe = gr.Blocks(title="AI Describe Image", css="#component-3, #component-5 {display: grid; align-content: center;}")

with describe:
    describe_tab = gr.TabItem(label='Describe')
    with describe_tab:
        input_column = gr.Row()
        with input_column:
            with gr.Column():
                input_image = grh.Image(label='Input', source='upload', type='numpy')
            with gr.Column():
                content_type = gr.Radio(
                    label='Content Type',
                    choices=[flags.desc_type_photo, flags.desc_type_anime],
                    value=flags.desc_type_photo
                )
                desc_btn = gr.Button(value='Describe this Image into Prompt')
                outputs=gr.Textbox(type="text", label="Output", show_copy_button=True)
    
    desc_btn.click(interrogatorFunction, inputs=[input_image, content_type], outputs=[outputs])

describe.launch()
