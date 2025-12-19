
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
        print("1. Start Demo Server (God Mode / WebRTC / API)")
        print("   -> Launches the FastAPI backend for the presentation.")
        print("")
        print("2. Train Model (V2 - Dual GPU Optimized)")
        print("   -> Starts the multi-GPU training pipeline for new styles.")
        print("")
        print("3. View Project Info / Credits")
        print("   -> Shows references and citations.")
        print("")
        print("4. Exit")
        print("===============================================================")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\n>> Starting Demo Server (src/server.py)...")
            server_path = os.path.join("src", "server.py")
            if os.path.exists(server_path):
                os.system(f"python {server_path}")
            else:
                print(f"[ERROR] {server_path} not found. Have you implemented the backend yet?")
                input("\nPress Enter to return to menu...")
                
        elif choice == '2':
            print("\n>> Starting V2 Training (src/train_v2.py)...")
            train_path = os.path.join("src", "train_v2.py")
            if os.path.exists(train_path):
                # Ensure we are running from project root
                os.system(f"python {train_path}")
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
