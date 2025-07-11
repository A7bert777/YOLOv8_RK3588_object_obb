#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "yolov8-obb.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include <cstring> // For strcmp  
#include <opencv2/opencv.hpp>  
#include <iostream>
#include <string> 
#include <chrono>  
#include <vector> 

// 检查是否是图片文件的函数
bool is_image_file(const char *filename) 
{
    const char *ext = strrchr(filename, '.');
    if (!ext) return false;
    
    return (strcasecmp(ext, ".jpg") == 0 ||
            strcasecmp(ext, ".jpeg") == 0 ||
            strcasecmp(ext, ".png") == 0 ||
            strcasecmp(ext, ".bmp") == 0);
}

int write_image_mine(const char* path, const image_buffer_t* img) 
{  
    int width = img->width;  
    int height = img->height;  
    int channels = (img->format == IMAGE_FORMAT_RGB888) ? 3 :   
                   (img->format == IMAGE_FORMAT_GRAY8) ? 1 :   
                   4; // 根据image_buffer_t中的format字段确定通道数  
    void* data = img->virt_addr;  
  
    // 假设图像数据是连续的，且每个通道的数据类型是8位无符号整数  
    cv::Mat cv_img(height, width, CV_8UC(channels), data);  
    if (channels == 3) 
    {
        cv::Mat bgr_img;  
        cv::cvtColor(cv_img, bgr_img, cv::COLOR_RGB2BGR);  
        bool success = cv::imwrite(path, bgr_img);  
        return success ? 0 : -1;  
    }  
  
    // 其他直接保存  
    bool success = cv::imwrite(path, cv_img);  
    return success ? 0 : -1; // 成功返回0，失败返回-1  
}
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    // 设置默认路径
    const char *default_model_path = "/home/firefly/yolov8obb/model/car_100_best_300epoch_relu.rknn";
    const char *default_input_dir = "/home/firefly/yolov8obb/inputimage";
    const char *default_output_dir = "/home/firefly/yolov8obb/outputimage";

    const char *model_path = default_model_path;
    const char *input_dir = default_input_dir;
    const char *output_dir = default_output_dir;

    // 如果提供了命令行参数，则覆盖默认值
    if (argc == 4) 
    {
        model_path = argv[1];
        input_dir = argv[2];
        output_dir = argv[3];
    } 
    else if (argc != 1) {  // 既不是无参数也不是3个参数
        printf("Usage: %s [<model_path> <input_dir> <output_dir>]\n", argv[0]);
        printf("Using default paths when no arguments provided.\n");
        printf("Defaults:\n  Model: %s\n  Input: %s\n  Output: %s\n", 
               default_model_path, default_input_dir, default_output_dir);
        return -1;
    }

    printf("Using configuration:\n");
    printf("  Model path: %s\n", model_path);
    printf("  Input directory: %s\n", input_dir);
    printf("  Output directory: %s\n\n", output_dir);

    // 创建输出目录（如果不存在）
    mkdir(output_dir, 0755);

    int ret = 0;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    DIR *dir = NULL;

    init_post_process();

    ret = init_yolov8_obb_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_obb_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    dir = opendir(input_dir);
    if (!dir) {
        perror("opendir");
        ret = -1;
        goto out;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) 
    {
        if (entry->d_type != DT_REG) continue;
        
        const char *filename = entry->d_name;
        if (!is_image_file(filename)) continue;

        char input_path[1024];
        char output_path[1024];
        snprintf(input_path, sizeof(input_path), "%s/%s", input_dir, filename);
        snprintf(output_path, sizeof(output_path), "%s/%s", output_dir, filename);

        printf("\nProcessing: %s\n", input_path);

        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        ret = read_image(input_path, &src_image);

        if (ret != 0)
        {
            printf("read image fail! ret=%d image_path=%s\n", ret, input_path);
            if (src_image.virt_addr != NULL) {
                free(src_image.virt_addr);
            }
            continue;
        }

        object_detect_result_list od_results;
        memset(&od_results, 0, sizeof(object_detect_result_list));

        ret = inference_yolov8_obb_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("inference_yolov8_obb_model fail! ret=%d\n", ret);
            free(src_image.virt_addr);
            continue;
        }

        char text[256];
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d angle=%f) %.3f\n", coco_cls_to_name(det_result->cls_id),
                   det_result->box.x, det_result->box.y,
                   det_result->box.w, det_result->box.h,
                   det_result->box.angle, det_result->prop);
            int x1 = det_result->box.x;
            int y1 = det_result->box.y;
            int w = det_result->box.w;
            int h = det_result->box.h;
            float angle = det_result->box.angle;

            draw_obb_rectangle(&src_image, x1, y1, w, h, angle, COLOR_BLUE, 3);

            snprintf(text, sizeof(text), "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 12);
        }

        //write_image(output_path, &src_image);
        write_image_mine(output_path, &src_image);
        printf("Saved result to: %s\n", output_path);

        free(src_image.virt_addr);
    }

out:
    if (dir) {
        closedir(dir);
    }
    
    deinit_post_process();

    ret = release_yolov8_obb_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov8_obb_model fail! ret=%d\n", ret);
    }

    return ret < 0 ? -1 : 0;
}