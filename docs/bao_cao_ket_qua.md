# Báo cáo kết quả: Nhận diện rác thải sinh hoạt 9 lớp với YOLOv8

Tài liệu tổng hợp dữ liệu, quy trình kiểm tra nhãn, cấu hình huấn luyện, kết quả đánh giá và giới hạn của các mô hình `waste_v2`, `waste_v3` và `waste_v5` trong dự án Waste Sorting Vision. Mọi số liệu đều được đo lại trên máy bằng Ultralytics 8.4.37, `imgsz=640`, ngưỡng mặc định của `model.val()` (conf 0,001, IoU NMS 0,7).

## 1. Tóm tắt

- Mô hình cuối cùng là **YOLOv8m** (`models/waste_v5.pt`), nhận diện 9 lớp rác thải: Paper, Paper Cup, Vinyl, Plastic, Glass, Can, PET, Styrofoam, Battery.
- Trên tập validation `waste_v5` (3.484 ảnh), mô hình đạt **mAP@50 = 0,768** và **mAP@50-95 = 0,644**, cao hơn hai phiên bản YOLOv8n trước đó (0,677 và 0,699).
- Trên ảnh rác thực tế ngoài trời (tập validation chính thức của TACO, 208 ảnh), mô hình đạt **mAP@50 = 0,381**. Kết quả thấp hơn nhiều so với tập validation tổng hợp, cho thấy bài toán nhận diện rác trong cảnh phức tạp vẫn còn khó.
- Trước lần huấn luyện cuối, toàn bộ nhãn được kiểm tra bằng mô hình. Bốn nguồn dữ liệu có nhãn sai hệ thống đã bị loại, 253 ảnh thiếu nhãn bị loại, 192 ảnh lệch hướng xoay (EXIF) được sửa.
- **Chưa có tập test độc lập.** Mọi số liệu trong báo cáo là số liệu trên tập validation (xem mục 8).

## 2. Bài toán và hệ lớp

Hệ 9 lớp được xây dựng lại từ hệ phân loại rác tái chế của Hàn Quốc (bộ AI-Hub dùng ở giai đoạn đầu dự án), rút gọn cho phù hợp với dữ liệu công khai hiện có:

| ID | Lớp | Bao gồm | Không bao gồm |
|---|---|---|---|
| 0 | Paper | Giấy, bìa carton, hộp giấy, vỏ hộp sữa giấy | Cốc giấy |
| 1 | Paper Cup | Cốc giấy dùng một lần | |
| 2 | Vinyl | Nhựa mềm dạng màng: túi nilon, vỏ bánh kẹo, màng bọc | |
| 3 | Plastic | Nhựa cứng: hộp, cốc nhựa, thìa dĩa, ống hút, **nắp chai** | Chai nước PET |
| 4 | Glass | Chai, lọ, cốc thủy tinh, mảnh kính vỡ | |
| 5 | Can | Lon nhôm, lon thiếc, bình xịt | |
| 6 | PET | Chai nhựa đựng đồ uống (thân chai) | Nắp chai |
| 7 | Styrofoam | Hộp xốp, cốc xốp, mảnh xốp | |
| 8 | Battery | Pin, ắc quy | |

So với các checkpoint cũ của dự án (15 và 16 lớp, huấn luyện trên AI-Hub), hệ 9 lớp không phân biệt màu thủy tinh và không tách rác sạch với rác bẩn, vì các nguồn dữ liệu công khai không có nhãn chi tiết này.

## 3. Dữ liệu

### 3.1. Nguồn dữ liệu và giấy phép

Tổng cộng 20 nguồn được sử dụng trong `waste_v5`. Số ảnh là số ảnh sau khi lọc, gộp và loại trùng.

| Nguồn | Liên kết | Giấy phép | Train | Val |
|---|---|---|---|---|
| TACO (official) | https://github.com/pedropro/TACO | Code MIT; ảnh từ Flickr (CC) và OpenLitterMap (ODbL), một phần không ghi giấy phép | 1.185 | 208 |
| TACO (unofficial, cộng đồng gán nhãn) | https://github.com/pedropro/TACO | Như trên | 2.599 | 472 |
| battery_detection | https://universe.roboflow.com/waste-seregation/battery-detection-m7onv/dataset/1 | CC BY 4.0 | 757 | 134 |
| pet_bottle_type | https://universe.roboflow.com/project-yysbf/pet-bottle-type/dataset/12 | Public Domain | 476 | 210 |
| paper_cup | https://universe.roboflow.com/project-zq0wz/paper-cup-xhrgq/dataset/3 | Public Domain | 240 | 160 |
| styrofoam | https://universe.roboflow.com/a-3zezt/styrofoam-wfuot/dataset/1 | CC BY 4.0 | 277 | 49 |
| soda_can | https://universe.roboflow.com/personal-g5mzf/soda-can-object-detection/dataset/3 | CC BY 4.0 | 71 | 29 |
| soda_can_small | https://universe.roboflow.com/cyberwarriorstemcamp/soda-can-qr4c4/dataset/2 | CC BY 4.0 | 71 | 39 |
| waste_mixed | https://universe.roboflow.com/waste-classification-3zmlr/waste-hsysm-xnrsr/dataset/1 | CC BY 4.0 | 1.037 | 456 |
| plastic_bag | https://universe.roboflow.com/ecosorter-drvgm/plastic-bag-dataset/dataset/6 | CC BY 4.0 | 329 | 21 |
| plastic_bags_small | https://universe.roboflow.com/dataset-t7hz7/plastic-bags-0qzjp/dataset/3 | CC BY 4.0 | 161 | 102 |
| food_wrappers | https://universe.roboflow.com/aia-assignment-4xyx1/food-wrappers-dpjkz-fer1g/dataset/1 | CC BY 4.0 | 212 | 38 |
| plastic_waste_mgmt | https://universe.roboflow.com/yolov5-6agzx/plastic-waste-management-6p8kw/dataset/15 | CC BY 4.0 | 1.595 | 517 |
| litr_drinking | https://universe.roboflow.com/k-s/litr-drinking-waste/dataset/2 | CC BY 4.0 | 710 | 290 |
| trash_plastic_bottle | https://universe.roboflow.com/ros/trash-plastic-bottle-detection/dataset/2 | CC BY 4.0 | 54 | 21 |
| plastic_bottles | https://universe.roboflow.com/waste-rq8p9/plastic-bottles-uu8v9/dataset/1 | CC BY 4.0 | 211 | 89 |
| river_trash | https://universe.roboflow.com/trisha-lingat-m9vjl/river-trash-final-k9997/dataset/2 | CC BY 4.0 | 412 | 188 |
| beach_litter | https://universe.roboflow.com/beach-litter-jgn6d/beach-litter-wt1od/dataset/1 | CC BY 4.0 | 16 | 6 |
| bottle_can_pack | https://universe.roboflow.com/recyclorobloai-intern/can-bottle-and-pack-detection/dataset/1 | CC BY 4.0 | 550 | 138 |
| bottles_and_cans | https://universe.roboflow.com/raj-sangani-ygq8p/bottles-and-cans/dataset/1 | CC BY 4.0 | 639 | 161 |
| recycling_glass | https://universe.roboflow.com/recyclestuff/updated-recycling-dataset/dataset/9 | CC BY 4.0 | 245 | 156 |
| **Tổng** | | | **11.847** | **3.484** |

Giấy phép CC BY 4.0 yêu cầu ghi nguồn khi sử dụng lại, và bảng trên đáp ứng yêu cầu này. Ảnh TACO có giấy phép không đồng nhất; nếu công bố lại ảnh TACO, cần kiểm tra giấy phép của từng ảnh.

### 3.2. Phân bố lớp

Số box (đối tượng) sau khi gộp, trước khi lặp ảnh:

| Lớp | Train | Val |
|---|---|---|
| Paper | 1.879 | 712 |
| Paper Cup | 973 | 250 |
| Vinyl | 3.637 | 612 |
| Plastic | 3.655 | 733 |
| Glass | 961 | 325 |
| Can | 1.624 | 667 |
| PET | 3.806 | 1.825 |
| Styrofoam | 667 | 142 |
| Battery | 1.480 | 258 |

Glass và Styrofoam là hai lớp ít dữ liệu nhất. Ảnh train chứa hai lớp này được lặp lại 2 lần (repeat-factor sampling), thêm 1.199 ảnh, nâng tổng số ảnh train đưa vào huấn luyện lên 13.046. Tập validation không bị lặp.

### 3.3. Xử lý dữ liệu

1. **Chuyển đổi và gộp nhãn** (`scripts/convert_taco_to_yolo.py`, `scripts/convert_taco_unofficial.py`, `scripts/merge_yolo_datasets.py`): chuyển nhãn COCO và nhãn đa giác sang box YOLO, ánh xạ tên lớp của từng nguồn về 9 lớp.
2. **Bỏ ảnh có vật thuộc lớp mơ hồ.** Ví dụ lớp "cup" (không rõ giấy hay nhựa) hay "Unlabeled litter" của TACO. Nếu giữ lại, những vật này sẽ trở thành "nền" và dạy mô hình sai. Các lớp không phải rác như "branch" hay "oil-spill" chỉ bị bỏ box, ảnh vẫn giữ.
3. **Giữ một bản cho mỗi ảnh gốc.** Các export của Roboflow chứa nhiều bản augment của cùng một ảnh; các bản của cùng một ảnh gốc không bao giờ nằm ở cả train lẫn val.
4. **Loại ảnh gần trùng** bằng difference hash 256 bit (ngưỡng ≤ 10 bit khác nhau). Kết quả:
   - `litr_drinking` chứa 141 ảnh gốc của TACO.
   - `bottles_and_cans` có 716 ảnh trùng với nguồn khác.
   - `recycling_glass` có 1.418 ảnh trùng với `waste_mixed`.
   - Tất cả đều đã bị loại. Nếu không loại, ảnh validation TACO sẽ lọt vào tập train.
5. **Sửa hướng xoay ảnh (EXIF).** 192 ảnh TACO unofficial có nhãn vẽ theo ảnh chưa xoay nhưng file ảnh mang cờ xoay EXIF. Nếu để nguyên, Ultralytics sẽ xoay ảnh khi huấn luyện và mọi box sẽ lệch khỏi vật. Các ảnh này được ghi lại theo đúng hướng của nhãn. 8 ảnh có kích thước không khớp nhãn bị loại.
6. **Đóng gói** (`scripts/package_dataset.py`): thu nhỏ ảnh về cạnh dài tối đa 1.280 px (nhãn YOLO dùng tọa độ chuẩn hóa nên không đổi), rồi kiểm tra từng ảnh có đúng một file nhãn.

## 4. Quy trình kiểm tra nhãn

### 4.1. Phương pháp

Script `scripts/audit_dataset.py` dùng mô hình đã huấn luyện (`waste_v3`) chạy lại trên toàn bộ khoảng 13.700 ảnh train và val, rồi so sánh dự đoán với nhãn. Mỗi ảnh được đánh giá theo ba chỉ số:

- **Thiếu nhãn:** mô hình phát hiện vật với độ tin cậy ≥ 0,6 nhưng vị trí đó không có nhãn nào (IoU < 0,3).
- **Khác lớp:** nhãn và dự đoán tin cậy trùng vị trí (IoU ≥ 0,5) nhưng khác lớp.
- **Không phát hiện:** có nhãn nhưng mô hình không phát hiện vật nào ở vị trí đó.

Ngoài ra script còn phát hiện box quá nhỏ (dưới 8 px), box phủ gần hết ảnh và box trùng lặp. Các ảnh có điểm tệ nhất của mỗi nguồn được xem trực tiếp để phân biệt nhãn sai với lỗi của mô hình.

### 4.2. Nguồn bị loại

| Nguồn | Lý do |
|---|---|
| trash_detection2 | Bản sao của TACO trên Roboflow; gây trùng ảnh và rò rỉ tập validation |
| garbage_cls3 | Giá sách được gán nhãn Paper (hơn 100 box một ảnh), lá cây xanh được gán nhãn Glass |
| beverage_containers | Ly bia, ly rượu trên bàn ăn (không phải rác), nhiều vật không có nhãn, còn ô đen do augmentation |
| aluminium_cans | Đống lon chỉ gán nhãn một phần (23,8% ảnh thiếu nhãn) |
| plastic_litter | Ảnh drone, vật chỉ 2–5 px; 66% box mô hình không thể phát hiện |
| vn_drinks, soda_bottles | Ảnh kệ hàng và tủ lạnh, phần lớn chai không có nhãn |
| waste_in_water, yolo_waste | Nhãn sai (lon gán nhãn "Plastic Bottle", ống tuýp gán nhãn "Aluminum can") |

### 4.3. Ảnh bị loại và lọc có điều kiện

- **253 ảnh thiếu nhãn** bị loại khỏi các nguồn không phải TACO, danh sách lưu tại `configs/audit_exclude.txt`. Nhiều nhất từ paper_cup (55), plastic_bottles (33), plastic_waste_mgmt (30), litr_drinking (29), battery_detection (26).
- **river_trash:** chỉ giữ ảnh có tối đa 5 box. Ảnh bãi rác lớn chỉ gán nhãn vài vật trong hàng trăm vật.
- **plastic_bottles:** loại ảnh có box phủ trên 60% diện tích, tức cả đống chai bị khoanh thành một box.
- TACO chính thức không bị lọc, để giữ nguyên tập validation dùng làm thước đo cố định.

## 5. Huấn luyện

| Phiên bản | Mô hình | Dữ liệu | Cấu hình | Phần cứng |
|---|---|---|---|---|
| waste_v2 | YOLOv8n | TACO và 11 nguồn Roboflow (7.058 ảnh train) | 640 px, 100 epoch, batch 16 | Colab T4 |
| (thử nghiệm) | YOLOv8s | Như waste_v2 | 1.024 px, 60 epoch, batch 8 | Colab A100 |
| waste_v3 | YOLOv8n | Thêm ảnh rác thực tế (8.561 ảnh train) | 640 px, 100 epoch, batch 32 | Colab A100 |
| **waste_v5** | **YOLOv8m** | Dữ liệu đã kiểm tra nhãn (13.046 ảnh train) | 640 px, tối đa 150 epoch, patience 30, batch 32 | Colab A100 |

Mọi phiên bản dùng trọng số khởi tạo từ COCO (`yolov8*.pt`) và augmentation mặc định của Ultralytics. `waste_v5` dừng sớm ở epoch 92; trọng số tốt nhất ở epoch 62.

Bản thử nghiệm YOLOv8s ở 1.024 px không tốt hơn YOLOv8n ở 640 px (0,760 so với 0,779 trên tập val của waste_v2) mà chạy chậm hơn khoảng 5 lần. Nguyên nhân: ảnh TACO chỉ có bản 640 px trên Flickr, nên tăng kích thước đầu vào không bổ sung chi tiết.

## 6. Kết quả

### 6.1. So sánh các phiên bản

Cả ba mô hình được đánh giá trên cùng các tập ảnh của `waste_v5`:

| Tập đánh giá | Số ảnh | v2 (YOLOv8n) | v3 (YOLOv8n) | **v5 (YOLOv8m)** |
|---|---|---|---|---|
| Val đầy đủ, mAP@50 | 3.484 | 0,677 | 0,699 | **0,768** |
| Val đầy đủ, mAP@50-95 | 3.484 | 0,551 | 0,575 | **0,644** |
| Val đã loại ảnh gần trùng với train, mAP@50 | 3.415 | 0,671 | 0,693 | **0,765** |
| TACO chính thức, mAP@50 | 208 | 0,362 | 0,283 | **0,381** |
| TACO chính thức, mAP@50-95 | 208 | 0,289 | 0,227 | **0,324** |
| TACO chính thức và unofficial, mAP@50 | 680 | 0,210 | 0,195 | **0,268** |

Tập TACO chính thức là thước đo công bằng nhất giữa ba mô hình, vì không mô hình nào được huấn luyện trên nó. Với phần TACO unofficial, `waste_v5` đã được huấn luyện trên phần train của cùng nguồn, còn v2 và v3 thì chưa, nên chênh lệch ở dòng cuối một phần đến từ việc v5 đã quen với kiểu ảnh này.

### 6.2. Theo từng lớp (mAP@50)

| Lớp | Val đầy đủ: v2 | v3 | **v5** | TACO chính thức: v2 | v3 | **v5** |
|---|---|---|---|---|---|---|
| Paper | 0,645 | 0,693 | **0,817** | 0,296 | 0,260 | **0,300** |
| Paper Cup | 0,785 | 0,776 | **0,790** | **0,335** | 0,319 | 0,298 |
| Vinyl | 0,562 | 0,600 | **0,664** | 0,362 | 0,313 | **0,428** |
| Plastic | 0,577 | 0,607 | **0,664** | 0,275 | **0,297** | 0,286 |
| Glass | 0,621 | 0,659 | **0,771** | **0,110** | 0,091 | 0,077 |
| Can | 0,737 | 0,687 | **0,864** | 0,801 | 0,688 | **0,806** |
| PET | 0,724 | 0,811 | **0,897** | 0,544 | 0,403 | **0,722** |
| Styrofoam | 0,539 | **0,563** | 0,559 | **0,203** | 0,148 | 0,184 |
| Battery | **0,904** | 0,896 | 0,885 | 0,332 | 0,030 | 0,332 |

Tập TACO chính thức chỉ có 1 box Battery, 12 box Paper Cup, 25 box Can và 30 box Styrofoam, nên các con số của những lớp này dao động mạnh.

### 6.3. Tốc độ

| Mô hình | Tham số | Suy luận trên Apple M1 (GPU, MPS) | Suy luận trên CPU, 1 ảnh |
|---|---|---|---|
| YOLOv8n (v2, v3) | 3,0 triệu | khoảng 7–12 ms/ảnh | khoảng 40–45 ms |
| YOLOv8m (v5) | 25,9 triệu | khoảng 68 ms/ảnh | khoảng 183 ms |

### 6.4. Checkpoint cũ 15 và 16 lớp

Hai checkpoint `best.pt` (15 lớp) và `best5.pt` (16 lớp), huấn luyện trên ảnh băng chuyền AI-Hub, được đánh giá trên tập validation của `waste_v2` sau khi quy đổi lớp về 9 lớp. Kết quả mAP@50 chỉ đạt 0,030 và 0,086. Tuy vậy, trên ảnh băng chuyền của AI-Hub, hai mô hình này vẫn hoạt động tốt, cho thấy khoảng cách miền dữ liệu (domain gap) rất lớn giữa ảnh băng chuyền và ảnh rác ngoài đời.

## 7. Phân tích lỗi

**Chai nhựa in nhãn bị nhận là lon (waste_v3).** Trên ảnh chai nhựa có nhãn in màu bọc kín thân, v3 dự đoán **Can với độ tin cậy 0,90**, trong khi v2 dự đoán đúng PET 0,82.
- Nguyên nhân: dữ liệu PET thêm vào ở v3 chủ yếu là chai trong suốt, còn dữ liệu Can có nhiều lon in thương hiệu. Mô hình học quy tắc sai "hình trụ in màu là lon".
- Khắc phục ở v5: thêm nguồn `bottle_can_pack`, có chai nhựa in nhãn và lon chụp cùng khung. Kết quả PET trên TACO chính thức tăng từ 0,403 lên 0,722.
- Ảnh gốc của trường hợp này chưa được đánh giá lại với v5, vì chỉ có bản ảnh đã vẽ sẵn box.

**Pin không được nhận ra.** Với ảnh pin AA màu đen nằm ngang, bọc chung trong màng co, cả v2 và v3 đều không đưa ra dự đoán Battery nào, kể cả ở ngưỡng gần 0.
- Nguyên nhân: dữ liệu pin (battery_detection) chủ yếu là pin đứng thẳng, vỏ nhiều màu, pin laptop và ắc quy.
- Điểm Battery cao trên tập validation (0,89–0,90) phản ánh ảnh cùng nguồn với tập train, không phản ánh khả năng nhận pin nói chung.

**Hộp nhựa đục bị nhận là Styrofoam.** Hộp đựng pin bằng nhựa trắng đục có vách ngăn bị dự đoán là Styrofoam, vì trông giống hộp xốp đựng thức ăn. Dữ liệu train có rất ít hộp nhựa đục.

**Bỏ sót vật trong cảnh phức tạp.** Trên tập TACO chính thức, ở ngưỡng 0,25, cả v2 và v3 đều không phát hiện khoảng 50% số vật. Glass bị bỏ sót 40/47 vật. Nguyên nhân chính là vật nhỏ, bị che khuất và lẫn vào nền, chứ không phải do phân loại sai.

**Lỗi ứng dụng (đã sửa).** Ứng dụng Streamlit từng đưa ảnh tải lên vào mô hình theo thứ tự màu RGB trong khi Ultralytics xử lý mảng numpy theo thứ tự BGR. Kết quả là chai PET màu xanh ngọc bị nhận là Glass 0,78. Sau khi sửa, cùng ảnh đó được nhận là PET 0,83. Ứng dụng cũng bật NMS không phân biệt lớp (`agnostic_nms`) để một vật không bị vẽ thành hai nhãn khác nhau. Lỗi này không ảnh hưởng đến các số liệu mAP, vì mAP được đo trực tiếp bằng Ultralytics.

## 8. Giới hạn

1. **Chưa có tập test độc lập.** Tập validation đã được dùng để chọn epoch tốt nhất và quyết định dừng sớm, nên các số liệu có phần lạc quan. Dự kiến bổ sung 30–50 ảnh rác chụp tại Việt Nam, đánh giá bằng `scripts/evaluate_test_set.py`.
2. **Tập đánh giá ảnh thực tế nhỏ.** TACO chính thức chỉ có 208 ảnh; một số lớp chỉ có 1–30 box. Chênh lệch mAP khoảng ±0,01–0,03 trên tập này có thể là dao động ngẫu nhiên.
3. **Các phiên bản khác nhau cả mô hình lẫn dữ liệu.** v2, v3 dùng YOLOv8n còn v5 dùng YOLOv8m và dữ liệu mới, nên không tách được riêng ảnh hưởng của từng yếu tố.
4. **Mỗi cấu hình chỉ huấn luyện một lần.** Chưa lặp lại với seed khác, nên không có khoảng tin cậy cho kết quả.
5. **Nhãn còn nhiễu.** Việc kiểm tra nhãn dựa vào một mô hình, nên chỉ phát hiện được lỗi mà mô hình nhận ra. Một số nguồn vẫn còn ảnh thiếu nhãn hoặc nhãn không nhất quán, ví dụ quy ước chai nhựa không phải chai nước giữa TACO và các nguồn khác.
6. **Glass và Styrofoam vẫn yếu trên ảnh thực tế** (mAP@50 khoảng 0,08–0,18 trên TACO), do thiếu dữ liệu công khai về chai thủy tinh và hộp xốp là rác ngoài đời.
7. **Dữ liệu chủ yếu từ nước ngoài.** Chưa có dữ liệu rác tại Việt Nam, với bao bì và nhãn hiệu địa phương.
8. **Ảnh TACO có độ phân giải thấp** (bản 640 px từ Flickr), nên không khai thác được lợi ích của kích thước đầu vào lớn hơn.
9. **Tốc độ.** YOLOv8m chạy trên CPU khoảng 183 ms/ảnh, phù hợp với ảnh tĩnh nhưng kém mượt với video và webcam. Có thể chọn checkpoint YOLOv8n trong ứng dụng khi cần tốc độ.

## 9. Hướng phát triển

- Xây dựng tập test ảnh rác tại Việt Nam và báo cáo kết quả trên tập này.
- Thu thập thêm ảnh Glass, Styrofoam và Battery trong bối cảnh rác thực tế: pin nằm ngang, pin trong vỉ, hộp nhựa đục.
- Tải ảnh TACO ở độ phân giải gốc để thử huấn luyện ở 1.024 px.
- Chạy lặp nhiều seed và so sánh các cỡ YOLOv8 (n, s, m) trên cùng một bộ dữ liệu.

## 10. Tái lập kết quả

```bash
# 1. Tải dữ liệu
python scripts/download_taco.py
python scripts/download_taco.py annotations_unofficial.json
ROBOFLOW_API_KEY=<key> python scripts/download_roboflow.py

# 2. Chuyển đổi, gộp, đóng gói
python scripts/convert_taco_to_yolo.py
python scripts/convert_taco_unofficial.py
python scripts/merge_yolo_datasets.py      # dùng configs/audit_exclude.txt
python scripts/package_dataset.py waste_v5

# 3. Huấn luyện: notebooks/train_waste_v1_colab.ipynb (Colab, A100)

# 4. Kiểm tra nhãn và đánh giá tập test
python scripts/audit_dataset.py --model models/waste_v3.pt
python scripts/evaluate_test_set.py datasets/test_vn
```

`convert_taco_unofficial.py` tự sửa hướng xoay EXIF, thu nhỏ ảnh và bỏ 8 ảnh có kích thước không khớp nhãn. Chạy lại script cho cùng kết quả.
