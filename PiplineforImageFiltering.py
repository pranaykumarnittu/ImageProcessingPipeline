import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_fourier_transform(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    f_transform = np.fft.fft2(gray_frame)

    f_shift = np.fft.fftshift(f_transform)

    return f_shift

def inverse_fourier_transform(f_shift_filtered):
    f_ishift = np.fft.ifftshift(f_shift_filtered)

    img_back = np.fft.ifft2(f_ishift)

    img_back = np.abs(img_back)

    return img_back

def apply_filter(f_shift, filter_type='low-pass', radius=30):
    rows, cols = f_shift.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros_like(f_shift)

    if filter_type == 'low-pass':       
        mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1
    elif filter_type == 'high-pass':
        mask[:crow-radius, :] = 1
        mask[crow+radius:, :] = 1
        mask[:, :ccol-radius] = 1
        mask[:, ccol+radius:] = 1

    f_shift_filtered = f_shift * mask

    return f_shift_filtered

def normalize_image(image):
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    norm_image = np.uint8(norm_image)

    return norm_image

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        f_shift = apply_fourier_transform(frame)

        filter_type = 'low-pass'
        #filter_type = 'high-pass'

        f_shift_filtered = apply_filter(f_shift, filter_type=filter_type, radius=50)

        img_back = inverse_fourier_transform(f_shift_filtered)

        filtered_image = normalize_image(img_back)

        combined_frame = np.hstack((frame, cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)))

        cv2.imshow('Original (Left) | Filtered (Right)', combined_frame)

        if cv2.getWindowProperty('Original (Left) | Filtered (Right)', cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    exit(0)

if __name__ == "__main__":
    main()

    filter_radius = np.linspace(0, 100, 100)
    response = 60 * np.exp(-filter_radius / 20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(filter_radius, response, marker='*', color='b')

    ax.set_xlabel('Filter Radius', fontsize=12)
    ax.set_ylabel('Frequency Response', fontsize=12)
    ax.set_title('Frequency Response vs. Filter Radius', fontsize=14)

    ax.grid(True)

    plt.show()

