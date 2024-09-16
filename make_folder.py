import os
import shutil
import random

# Đường dẫn đến thư mục chứa hình ảnh
source_dir = './data_image/images/'  # Thay đổi thành đường dẫn thực tế của bạn

# Đường dẫn đến các thư mục đích
train_dir = './data_image/images/train'
val_dir = './data_image/images/val'
test_dir = './data_image/images/test'

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Lấy danh sách tất cả các tệp tin hình ảnh trong thư mục nguồn
all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Xáo trộn danh sách hình ảnh
random.shuffle(all_images)

# Xác định số lượng hình ảnh cho từng thư mục
total_images = len(all_images)
train_end = int(total_images * 0.70)
val_end = train_end + int(total_images * 0.15)

# Phân chia hình ảnh thành các nhóm train, val, test
train_images = all_images[:train_end]
val_images = all_images[train_end:val_end]
test_images = all_images[val_end:]

# Di chuyển các hình ảnh vào các thư mục tương ứng
for img in train_images:
    shutil.move(os.path.join(source_dir, img), os.path.join(train_dir, img))

for img in val_images:
    shutil.move(os.path.join(source_dir, img), os.path.join(val_dir, img))

for img in test_images:
    shutil.move(os.path.join(source_dir, img), os.path.join(test_dir, img))

print(f"Đã phân chia {total_images} hình ảnh thành các thư mục train, val, test.")
