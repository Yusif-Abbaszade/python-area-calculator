import cv2
import numpy as np
from tkinter import Tk, filedialog
from PIL import Image
import os

# Şəkil çəkilməsi üçün funksiya
def capture_image():
    # Kamera açılır
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        # Şəkli ekranda göstəririk
        cv2.imshow("Press 'c' to capture", frame)
       
        # 'c' düyməsi ilə şəkil çəkməyi təsdiqləyirik
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.imwrite("captured_image.jpg", frame)
            break
        elif key == ord('q'):
            break
   
    # Kamera bağlanır
    cap.release()
    cv2.destroyAllWindows()

# Fayldan şəkil seçmək
def select_image():
    Tk().withdraw()  # Tkinter pəncərəsini gizləyirik
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.png")])
   
    # Faylın yolunu göstəririk
    print(f"Seçilmiş şəkil yolu: {file_path}")
   
    # Faylın mövcudluğunu yoxlayırıq
    if not os.path.exists(file_path):
        print(f"Fayl tapılmadı: {file_path}")
        return None
    return file_path

# Ağ fonu çıxarmaq və sahəni hesablamaq
def remove_white_background_and_calculate_area(image_path):
    # Şəkli oxuyuruq
    image = cv2.imread(image_path)
   
    # Şəkli RGB formatına çeviririk
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
   
    # Ağ fonu çıxarmaq üçün ağ rəngin müəyyən edilmesi
    # Şəkildəki ağ rəngin dəyərləri
    lower_white = np.array([0, 0, 0])
    upper_white = np.array([100, 100, 100])

    # Ağ fonu maskalama
    mask = cv2.inRange(image_rgb, lower_white, upper_white)
    
    # Maskanı şəkildən çıxarırıq
    result = cv2.bitwise_and(image_rgb, image_rgb, mask=~mask)
   
    # Siyah ağ fonlu bölgənin sahəsini hesablamaq
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    area = cv2.countNonZero(threshold)
   
    # Şəkli göstəririk
    result_image = Image.fromarray(result)
    result_image.show()

    return area

# Əsas funksiya
def main():
    print("Proqram başlamalıdır.")
   
    # Şəkil seçmək (kamera və ya fayldan)
    while (True):
        option = input("Kame☻ra ilə şəkil çəkmək (1) və ya fayldan şəkil seçmək (2): ")
        if(option in ["1", "2"]):
            break
    if option == "1":
        capture_image()
        image_path = "captured_image.jpg"
    elif option == "2":
        image_path = select_image()
        if image_path is None:
            return  # Fayl tapılmadıqda proqram dayansın
    else:
        print("Yanlış seçim.")
        return

    # Şəkildən ağ fonu çıxarırıq və sahəni hesablayırıq
    area = remove_white_background_and_calculate_area(image_path)
    print(f"Qalan hissənin sahəsi: {area} piksel.")

if __name__ == "__main__":
    main()