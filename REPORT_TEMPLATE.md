# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

Ngày chạy:11/9/20262026

Runtime Colab: CPU/GPU

Python / PyTorch / Ultralytics:

Checkpoint: `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

Thay đổi so với notebook nguồn: Không 


## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`468`, `cab`, `1`, `510915`, `ImageNet-1K`):
- Record cab mô tả toàn bộ ảnh ở mức độ phân loại ảnh (image-level),  Ở đây, mô hình dự đoán ảnh thuộc lớp cab với điểm tin cậy khoảng 0,51. Điểm score chỉ là mức độ mô hình tin vào dự đoán và không phải ground truth.
- Class list đến từ taxonomy/dataset mà checkpoint được huấn luyện trên.
- class_id   cho máy dễ xử lý. class_name  cho con người dễ đọc. taxonomy_name cho biết class đó thuộc hệ thống phân loại nào.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? 
Quy định chọn một lớp đại diện cho toàn ảnh dựa trên nội dung/chủ thể chính.
- Vì sao model score không phải ground truth?
Vì score chỉ thể hiện mức độ tin cậy của mô hình, không chứng minh dự đoán đúng hay sai. Lab cũng nhấn mạnh prediction không phải nhãn chuẩn.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

## 2. Phát hiện vật thể – lớp và box cho từng object

 Một record: Có thể chọn object `person` trong ảnh `kitchen`, với score khoảng 0.91, như thể hiện trên hình `detection_predictions.png`.

* Diễn giải vị trí box: Box bao quanh người đứng ở khu vực giữa-phải của ảnh `kitchen`.

* So sánh số prediction ở hai threshold:
  Ở `kitchen`, threshold 0.20 → 17 object, 0.35 → 12 object, 0.60 → 6 object.

* Độ bao phủ và reviewer: Threshold thấp làm tăng số prediction và tăng độ bao phủ, nhưng reviewer phải kiểm tra nhiều kết quả hơn. Threshold cao giảm workload nhưng có nguy cơ bỏ sót object.

* Quy tắc box chặt: Box phải bao phủ toàn bộ phần object nhìn thấy và sát biên object, hạn chế tối đa phần nền dư thừa.

* Object bị che khuất/cắt mép: Cần guideline quy định có gán phần nhìn thấy hay không; trường hợp khó xác định biên/lớp thì cần escalation cho reviewer.


## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json`, sample `kitchen`.

YOLO11n-seg thực hiện instance segmentation: mỗi vật thể được phát hiện có một `class_name`, `score`, `bbox_xyxy` và một polygon riêng. Trong dữ liệu `kitchen`, ví dụ `kitchen-001` được dự đoán là `person`, score `0.899318`, với bbox `[385.45, 66.44, 498.02, 348.58]` và polygon gồm 348 điểm. :contentReference[oaicite:0]{index=0}

### Polygon bổ sung gì so với box?

- `bbox_xyxy` chỉ mô tả hình chữ nhật bao quanh toàn bộ vật thể.
- `polygon_xy` mô tả đường biên của vùng vật thể, vì vậy giữ được hình dạng thực tế chi tiết hơn.
- Ví dụ với `kitchen-001`, bbox bao phủ vùng từ `(385.45, 66.44)` đến `(498.02, 348.58)`, trong khi polygon có 348 điểm biên để mô tả hình dạng người. :contentReference[oaicite:1]{index=1}
- Vì vậy polygon hữu ích khi cần biết chính xác pixel/vùng nào thuộc về object, nhưng cũng khó kiểm tra chất lượng hơn box vì phải đánh giá cả đường biên.

### `instance_id` dùng để làm gì?

`instance_id` dùng để phân biệt từng object prediction riêng biệt, kể cả khi chúng có cùng `class_name`.

Ví dụ trong dữ liệu có các instance được đánh số như `kitchen-001`, `kitchen-003`, `kitchen-007`; mỗi record có class và polygon riêng. `kitchen-001` là `person`, còn `kitchen-003` là `bowl`. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

Trong code, `instance_id` được tạo theo dạng:

`{sample_id}-{index + 1:03d}`

Do đó, đây là ID của instance trong kết quả dự đoán của sample, không phải `class_id` và cũng không phải ID định danh cố định của một người/vật thể xuyên suốt nhiều ảnh hoặc video.

### Đề xuất quy tắc biên mask

- Mask nên bao phủ phần vật thể có thể quan sát được, bám sát biên nhìn thấy thay vì mở rộng sang background.
- Không nên dùng bbox để thay thế cho polygon vì bbox có thể chứa nhiều pixel background.
- Với vật thể bị cắt bởi mép ảnh, chỉ đánh dấu phần vật thể thực sự nhìn thấy trong ảnh.
- Khi biên quá mờ hoặc không thể xác định rõ từ ảnh, không nên tự suy đoán quá mức.

### Với vùng mờ/tiếp xúc/che khuất

Cần có guideline thống nhất cho các trường hợp:

- Biên mờ: quy định có lấy theo biên nhìn thấy hay cho phép sai số nhỏ.
- Hai object tiếp xúc: cần xác định đường biên giữa hai instance, tránh gộp chúng thành một mask.
- Object bị che khuất: cần thống nhất chỉ mask phần nhìn thấy hay suy luận phần bị che khuất.
- Object bị cắt bởi mép ảnh: cần thống nhất cách xử lý phần bị nằm ngoài khung hình.
- Nếu reviewer không thể xác định biên một cách nhất quán, nên escalation cho reviewer cấp cao thay vì tự quyết định.

### Vì sao segmentation khó hơn detection?

Detection chỉ cần kiểm tra class + bbox + score, còn segmentation phải kiểm tra thêm toàn bộ đường biên polygon. Trong file, số điểm polygon thay đổi đáng kể giữa các object, ví dụ `kitchen-001` có 348 điểm, còn các instance khác có số điểm khác nhau. :contentReference[oaicite:4]{index=4}

Do đó, segmentation có nhiều trường hợp khó đánh giá hơn như biên vật thể không rõ, vật thể chồng lấn/tiếp xúc hoặc bị che khuất.

## 4. Vòng đời và kiểm tra chất lượng

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho toàn bộ ảnh (`class_id`, `class_name`) | Ảnh khó phân loại, nhiều đối tượng nhưng không rõ lớp chính, hoặc lớp giữa các category gần nhau | Gán đúng một class theo guideline; nếu không chắc thì đánh dấu cần review | Kiểm tra class có đúng với nội dung ảnh và taxonomy hay không; kiểm tra các trường hợp borderline |
| Phát hiện vật thể | Một record cho mỗi object: `class_id`, `class_name`, `score`, `bbox_xyxy` | Thiếu object, nhầm class, box quá rộng/quá hẹp, box không bao phủ object, object bị che khuất hoặc cắt mép | Vẽ/chỉnh bbox sát object và gán class; xử lý theo guideline đối với object bị che khuất/cắt mép | Kiểm tra class, số lượng object và vị trí/kích thước bbox; đặc biệt xem box có bao phủ đủ object nhưng không lấy quá nhiều background |
| Instance segmentation | Một record cho mỗi instance: `instance_id`, `class_id`, `class_name`, `bbox_xyxy`, `polygon_xy` | Polygon lệch biên, quá rộng/quá hẹp, gộp hai object, tách sai một object, biên mờ hoặc object bị che khuất | Tạo/chỉnh polygon theo phần object nhìn thấy; mỗi instance có polygon riêng; áp dụng guideline hoặc đánh dấu escalation khi biên không rõ | Kiểm tra class, số instance, polygon có bám đúng biên object không, các object tiếp xúc có bị gộp không và vùng che khuất có được xử lý nhất quán không |
### Nguyên tắc QC/rework

QC cần kiểm tra theo đúng loại ground truth của từng task. Với classification, trọng tâm là đúng class. Với detection, trọng tâm là class + số lượng object + bbox. Với instance segmentation, cần kiểm tra thêm ranh giới polygon của từng instance.

Khi phát hiện lỗi, annotator thực hiện rework theo guideline. Những trường hợp không thể xác định rõ từ ảnh, đặc biệt là biên mờ, che khuất hoặc hai object tiếp xúc nhau, cần được đánh dấu để reviewer/escalation quyết định thay vì tự suy đoán.

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không chia sẻ hoặc sử dụng dữ liệu ngoài phạm vi được giao.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Reviewer/người phụ trách.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
