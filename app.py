print("Wait..")

def test():
    import gradio as gr
    
    def analyze_text(text):
        # Lakukan analisis atau pemrosesan teks di sini
        result = f"Anda memasukkan teks: {text}"
        return result
    
    iface = gr.Interface(
        fn=analyze_text,
        inputs=gr.Textbox(),  # Menggunakan input textbox
        outputs="text"  # Menetapkan output ke tipe teks
    )

    iface.launch()

def process():
    import subprocess

    def uninstall_and_install_gradio(version):
        # Uninstall current Gradio
        uninstall_command = ["pip", "uninstall", "gradio", "-y"]
        subprocess.run(uninstall_command)
        
        # Install specific version of Gradio
        install_command = ["pip", "install", f"gradio=={version}"]
        subprocess.run(install_command)
        
    # Gantilah "3.41.2" dengan versi Gradio yang diinginkan
    desired_version = "3.41.2"
        
    # Periksa versi Gradio yang terinstal
    current_version_command = ["pip", "show", "gradio"]
    result = subprocess.run(current_version_command, capture_output=True, text=True)
    current_version = None
    
    if "Version" in result.stdout:
        current_version = result.stdout.split("Version:")[1].strip()
    
    # Cek dan lakukan uninstall dan install jika versi tidak sesuai
    if current_version != desired_version:
        uninstall_and_install_gradio(desired_version)
        print(f"Gradio has been updated to version {desired_version}")
    else:
        print(f"Gradio is already at version {desired_version}")
    
    python_script = "entry_with_update.py"
    
    # Argument yang ingin Anda tambahkan
    # additional_arguments = ["--in-browser", "--all-in-fp32", "--directml", "--debug-mode", "--multi-user", "--always-cpu", "--is-windows-embedded-python"]
    additional_arguments = ["--always-cpu"]
    
    # Gabungkan semua argumen
    PIP = ["pip", "install", "-r", "requirements.txt"]
    command = ["python", python_script] + additional_arguments
    
    # Jalankan skrip menggunakan subprocess
    subprocess.run(PIP)
    print("Installing..")
    
    subprocess.run(command)# Menjalankan file batch
    print("Running..")
    # subprocess.run([batch_file_path], shell=True)

process()