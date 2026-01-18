import csv
from datetime import datetime
import io # Required for inner CSV parsing

def format_and_rename_columns(input_filename="combined_file.csv", output_filename="combined_formatted_renamed.csv"):
    """
    Đọc một tệp CSV có định dạng đặc biệt, định dạng lại các cột thời gian,
    đổi tên các cột được chỉ định, sắp xếp dữ liệu theo cột 'Date'
    và ghi kết quả vào một tệp CSV mới.
    """
    # Các cột cần định dạng lại ngày tháng (dựa trên tên cột gốc SAU KHI xử lý cột thừa)
    time_columns_to_format = ["timeOpen", "timeClose", "timeHigh", "timeLow", "timestamp"]

    # Ánh xạ đổi tên cột: old_name -> new_name
    column_rename_map = {
        "timeClose": "Date",  # This will be our sort key
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "marketCap": "Marketcap"
    }

    rows_to_write = []

    try:
        with open(input_filename, mode='r', newline='', encoding='utf-8-sig') as infile:
            # Bước 1: Đọc tệp CSV với dấu phẩy làm dấu phân cách chính
            comma_reader = csv.reader(infile, delimiter=',')

            # Xử lý header
            try:
                header_line_parts = next(comma_reader)
            except StopIteration:
                print(f"Lỗi: Tệp '{input_filename}' trống hoặc không có header.")
                return

            if len(header_line_parts) < 2 or not header_line_parts[1]:
                print(f"Lỗi: Dòng header không có định dạng mong đợi. Dòng header: {header_line_parts}")
                return

            header_string_data = io.StringIO(header_line_parts[1])
            semicolon_reader_header = csv.reader(header_string_data, delimiter=';')
            original_header = next(semicolon_reader_header)

            header_to_write = list(original_header)
            for old_name, new_name in column_rename_map.items():
                try:
                    idx = header_to_write.index(old_name)
                    header_to_write[idx] = new_name
                except ValueError:
                    print(f"Cảnh báo: Không tìm thấy cột '{old_name}' trong header gốc ('{original_header}') để đổi tên thành '{new_name}'.")

            rows_to_write.append(header_to_write)

            column_indices_to_format = []
            for col_name_to_format in time_columns_to_format:
                try:
                    column_indices_to_format.append(original_header.index(col_name_to_format))
                except ValueError:
                    if col_name_to_format:
                        print(f"Cảnh báo: Không tìm thấy cột '{col_name_to_format}' (để định dạng thời gian) trong header gốc. Bỏ qua định dạng cho cột này.")

            if not column_indices_to_format and time_columns_to_format:
                print("Thông báo: Không có cột thời gian nào được chỉ định trong 'time_columns_to_format' được tìm thấy trong header gốc để định dạng.")
            elif not time_columns_to_format:
                 print("Thông báo: Danh sách 'time_columns_to_format' trống. Sẽ không có định dạng thời gian nào được thực hiện.")


            # Xử lý các dòng dữ liệu
            for line_number, row_parts in enumerate(comma_reader, start=1):
                if not row_parts:
                    print(f"Cảnh báo: Dòng {line_number+1} trong tệp gốc trống. Bỏ qua.")
                    continue
                if len(row_parts) < 2:
                    print(f"Cảnh báo: Dòng {line_number+1} trong tệp gốc có định dạng không mong đợi: {row_parts}. Bỏ qua.")
                    continue

                data_string_part = row_parts[1]
                data_string_io = io.StringIO(data_string_part)
                semicolon_reader_data = csv.reader(data_string_io, delimiter=';')

                try:
                    row_data_fields = next(semicolon_reader_data)
                except StopIteration:
                    print(f"Cảnh báo: Không có dữ liệu sau khi phân tách bằng dấu chấm phẩy ở dòng {line_number+1}. Phần dữ liệu: '{data_string_part}'. Bỏ qua.")
                    continue

                if len(row_data_fields) != len(original_header):
                    print(f"Cảnh báo: Dòng dữ liệu {line_number} (sau khi xử lý) có số cột ({len(row_data_fields)}) không khớp với header ({len(original_header)}). Hàng: {row_data_fields}. Cố gắng xử lý.")

                new_row = list(row_data_fields)
                for col_idx in column_indices_to_format:
                    if col_idx < len(new_row) and new_row[col_idx]:
                        try:
                            # Check if already formatted (e.g. from a previous run or different source)
                            datetime.strptime(new_row[col_idx], "%Y-%m-%d")
                            # If it parses, it's already in the target format, so skip re-formatting
                        except ValueError:
                            # If it doesn't parse as "YYYY-MM-DD", try parsing as ISO format
                            try:
                                dt_object = datetime.strptime(new_row[col_idx], "%Y-%m-%dT%H:%M:%S.%fZ")
                                new_row[col_idx] = dt_object.strftime("%Y-%m-%d")
                            except ValueError:
                                print(f"Cảnh báo: Không thể phân tích cú pháp ngày '{new_row[col_idx]}' ở dòng dữ liệu {line_number}, cột '{original_header[col_idx]}' (chỉ số {col_idx}). Giữ nguyên giá trị.")
                        except IndexError:
                             print(f"Cảnh báo: Lỗi chỉ mục khi truy cập cột '{original_header[col_idx]}' (chỉ số {col_idx}) ở dòng dữ liệu {line_number} do hàng không đủ dài. Hàng: {new_row}")
                rows_to_write.append(new_row)

        # Bước 2: Sắp xếp dữ liệu (sau khi đã đọc và định dạng)
        if len(rows_to_write) > 1: # Cần có header và ít nhất một dòng dữ liệu
            header_row_for_sorting = rows_to_write[0]
            data_rows_to_sort = rows_to_write[1:]

            # Tìm chỉ số của cột 'Date' trong header_to_write (header đã đổi tên)
            # Mặc định là cột đầu tiên nếu không tìm thấy, nhưng nên có cảnh báo
            date_column_name_for_sorting = "Date" # Tên cột mới sau khi đổi tên
            try:
                date_column_index_for_sorting = header_row_for_sorting.index(date_column_name_for_sorting)

                # Sắp xếp các dòng dữ liệu dựa trên cột 'Date'
                # Định dạng "YYYY-MM-DD" có thể sắp xếp trực tiếp dưới dạng chuỗi
                # Thêm kiểm tra để tránh lỗi nếu hàng bị thiếu cột hoặc giá trị ngày trống
                def sort_key_func(row):
                    if date_column_index_for_sorting < len(row) and row[date_column_index_for_sorting]:
                        return row[date_column_index_for_sorting]
                    return "" # Hoặc một giá trị mặc định rất sớm/muộn nếu muốn xử lý khác

                data_rows_to_sort.sort(key=sort_key_func)
                print(f"Dữ liệu đã được sắp xếp theo cột '{date_column_name_for_sorting}'.")

                # Gộp lại header và các dòng dữ liệu đã sắp xếp
                rows_to_write = [header_row_for_sorting] + data_rows_to_sort

            except ValueError:
                print(f"Cảnh báo: Không tìm thấy cột '{date_column_name_for_sorting}' trong header đã xử lý ('{header_row_for_sorting}') để thực hiện sắp xếp. Dữ liệu sẽ không được sắp xếp.")
            except Exception as e_sort:
                print(f"Lỗi không mong muốn trong quá trình sắp xếp: {e_sort}. Dữ liệu có thể không được sắp xếp.")


        # Bước 3: Ghi kết quả ra tệp mới
        with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter=';') # Sử dụng dấu chấm phẩy cho tệp output
            writer.writerows(rows_to_write)

        print(f"Đã định dạng, đổi tên cột và sắp xếp thành công. Tệp được lưu vào: {output_filename}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp '{input_filename}'.")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ví dụ tạo tệp combined_file.csv để kiểm thử (GIỐNG VỚI ĐỊNH DẠNG CỦA BẠN)
    # Dữ liệu mẫu này sẽ được sắp xếp theo timeClose (sau khi đổi tên thành Date)
    # Dòng 2 (2018-12-30) sẽ đứng trước dòng 1 (2018-12-31)

    format_and_rename_columns(input_filename="combined_file.csv", output_filename="combined_formatted_renamed.csv")

    # Kiểm tra nội dung tệp output (tùy chọn)
    print("\nNội dung tệp output (combined_formatted_renamed.csv):")
    try:
        with open("combined_formatted_renamed.csv", "r", encoding="utf-8") as f_out:
            for line in f_out:
                print(line.strip())
    except FileNotFoundError:
        print("Tệp output không được tạo.")