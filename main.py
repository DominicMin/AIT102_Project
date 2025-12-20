
import os
import sys
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    while True:
        clear_screen()
        print("===============================================================")
        print("       AIT102 Group Project: Real-time Style Transfer          ")
        print("===============================================================")
        print(" Student Group ID: [INSERT_GROUP_ID]")
        print(" Project Title:    High-Performance Style Transfer Server      ")
        print("===============================================================")
        print("\nMAIN MENU:")
        print("1. Start Full Stack Demo (One-Click Launch)")
        print("   -> Launches Backend (Port 8000) and Frontend (Port 3000) in new windows.")
        print("")
        print("2. Train New Style (Manager Mode)")
        print("   -> Specify a style name to invoke the training manager.")
        print("")
        print("3. View Project Info / Credits")
        print("   -> Shows references and citations.")
        print("")
        print("4. Exit")
        print("===============================================================")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\n>> Starting Unified Server & Frontend (One-Click Launch)...")
            
            # 1. Start Backend (New Window)
            server_path = os.path.join("src", "server.py")
            if os.path.exists(server_path):
                print(f"   [Backend] Launching {server_path}...")
                if os.name == 'nt':
                    os.system(f"start cmd /k python {server_path}")
                else: 
                    # Unix/Linux support (background)
                    os.system(f"python {server_path} &")
            else:
                print(f"[ERROR] {server_path} not found.")

            # 2. Start Frontend (New Window)
            frontend_dir = os.path.join("frontend")
            if os.path.exists(frontend_dir):
                print(f"   [Frontend] Launching npm run dev in {frontend_dir}...")
                if os.name == 'nt':
                    os.system(f"start cmd /k \"cd {frontend_dir} && npm run dev\"")
                else:
                    os.system(f"cd {frontend_dir} && npm run dev &")
            else:
                 print(f"[ERROR] Frontend directory not found.")
            
            print("\n>> Services launched in new windows!")
            print("   Backend: http://localhost:8000")
            print("   Frontend: http://localhost:3000/realtime")
            input("\nPress Enter to return to menu...")
                
        elif choice == '2':
            print("\n[WARNING] High-Performance Training Mode")
            print("This process is optimized for Dual RTX 3090 GPUs (48GB Total VRAM).")
            print("Running on weaker hardware may cause OOM errors or system instability.")
            confirm = input("Type 'yes' to confirm found adequate hardware: ").strip().lower()

            if confirm != 'yes':
                print("[!] Operation cancelled by user.")
                input("\nPress Enter to return to menu...")
            else:
                print("\n>> Starting Training Manager (src/train_manager.py)...")
                style_name = input("Enter style name (e.g., 'picasso' for styles/picasso.jpg): ").strip()
                
                if not style_name:
                    print("[!] Style name cannot be empty.")
                else:
                    train_path = os.path.join("src", "train_manager.py")
                    if os.path.exists(train_path):
                        cmd = f"python {train_path} --style {style_name}"
                        print(f"   Running: {cmd}")
                        os.system(cmd)
                    else:
                        print(f"[ERROR] {train_path} not found.")
                input("\nPress Enter to return to menu...")
            
        elif choice == '3':
            print("\nCREDITS & REFERENCES:")
            print("- Neural Style Transfer (Gatys et al.): https://arxiv.org/abs/1508.06576")
            print("- Perceptual Losses (Johnson et al.): https://arxiv.org/abs/1603.08155")
            print("- VGG16 Architecture: https://keras.io/api/applications/vgg16/")
            print("- TensorFlow Distributed Training: https://www.tensorflow.org/guide/distributed_training")
            print("\nThis project was developed by AIT102 Group [ID].")
            input("\nPress Enter to return to menu...")
            
        elif choice == '4':
            print("\nExiting...")
            sys.exit()
        else:
            print("\n[!] Invalid selection. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    # Ensure current directory is in python path
    sys.path.append(os.getcwd())
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram Interrupted by User.")
        sys.exit()
