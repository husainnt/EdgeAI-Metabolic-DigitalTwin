#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_log.h"

#define I2C_PORT               I2C_NUM_0
#define I2C_SDA_PIN            21
#define I2C_SCL_PIN            22
#define I2C_FREQ_HZ            100000

// Device addresses
#define MPU6500_ADDR           0x68
#define MAX30102_ADDR          0x57

// MPU6500 Registers
#define MPU_REG_PWR_MGMT_1     0x6B
#define MPU_REG_ACCEL_XOUT_H   0x3B
#define MPU_REG_WHO_AM_I       0x75

// MAX30102 Registers
#define MAX_REG_INTR_STATUS_1  0x00
#define MAX_REG_FIFO_WR_PTR    0x04
#define MAX_REG_OVF_COUNTER    0x05
#define MAX_REG_FIFO_RD_PTR    0x06
#define MAX_REG_FIFO_DATA      0x07
#define MAX_REG_FIFO_CONFIG    0x08
#define MAX_REG_MODE_CONFIG    0x09
#define MAX_REG_SPO2_CONFIG    0x0A
#define MAX_REG_LED1_PA        0x0C  // Red LED amplitude
#define MAX_REG_LED2_PA        0x0D  // IR LED amplitude
#define MAX_REG_PART_ID        0xFF

static const char *TAG = "SENSOR_BRINGUP";

static esp_err_t i2c_write_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t data) {
    uint8_t write_buf[2] = {reg_addr, data};
    return i2c_master_write_to_device(I2C_PORT, dev_addr, write_buf, sizeof(write_buf), pdMS_TO_TICKS(100));
}

static esp_err_t i2c_read_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, size_t len) {
    return i2c_master_write_read_device(I2C_PORT, dev_addr, &reg_addr, 1, data, len, pdMS_TO_TICKS(100));
}

static esp_err_t init_mpu6500(void) {
    uint8_t who_am_i = 0;
    ESP_ERROR_CHECK(i2c_read_reg(MPU6500_ADDR, MPU_REG_WHO_AM_I, &who_am_i, 1));
    ESP_LOGI(TAG, "MPU WHO_AM_I: 0x%02X (Expected: 0x70 for MPU6500, 0x68 for MPU6050)", who_am_i);

    // Wake up MPU6500 (clears sleep bit in PWR_MGMT_1)
    return i2c_write_reg(MPU6500_ADDR, MPU_REG_PWR_MGMT_1, 0x00);
}

static esp_err_t init_max30102(void) {
    uint8_t part_id = 0;
    ESP_ERROR_CHECK(i2c_read_reg(MAX30102_ADDR, MAX_REG_PART_ID, &part_id, 1));
    ESP_LOGI(TAG, "MAX30102 PART_ID: 0x%02X (Expected: 0x15)", part_id);

    // Reset module
    ESP_ERROR_CHECK(i2c_write_reg(MAX30102_ADDR, MAX_REG_MODE_CONFIG, 0x40));
    vTaskDelay(pdMS_TO_TICKS(50));

    // FIFO configuration (sample averaging: 4, FIFO rollover enabled)
    ESP_ERROR_CHECK(i2c_write_reg(MAX30102_ADDR, MAX_REG_FIFO_CONFIG, 0x5F));
    // SpO2 Mode (Red + IR)
    ESP_ERROR_CHECK(i2c_write_reg(MAX30102_ADDR, MAX_REG_MODE_CONFIG, 0x03));
    // SpO2 config (ADC range 4096nA, 100 samples/sec, 411us pulse width)
    ESP_ERROR_CHECK(i2c_write_reg(MAX30102_ADDR, MAX_REG_SPO2_CONFIG, 0x27));
    // LED currents (~6.2mA each)
    ESP_ERROR_CHECK(i2c_write_reg(MAX30102_ADDR, MAX_REG_LED1_PA, 0x1F));
    ESP_ERROR_CHECK(i2c_write_reg(MAX30102_ADDR, MAX_REG_LED2_PA, 0x1F));

    // Reset FIFO pointers
    ESP_ERROR_CHECK(i2c_write_reg(MAX30102_ADDR, MAX_REG_FIFO_WR_PTR, 0x00));
    ESP_ERROR_CHECK(i2c_write_reg(MAX30102_ADDR, MAX_REG_OVF_COUNTER, 0x00));
    return i2c_write_reg(MAX30102_ADDR, MAX_REG_FIFO_RD_PTR, 0x00);
}

void app_main(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_SDA_PIN,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_io_num = I2C_SCL_PIN,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_FREQ_HZ,
    };
    ESP_ERROR_CHECK(i2c_param_config(I2C_PORT, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_PORT, conf.mode, 0, 0, 0));

    ESP_LOGI(TAG, "Initializing Sensors...");
    ESP_ERROR_CHECK(init_mpu6500());
    ESP_ERROR_CHECK(init_max30102());
    ESP_LOGI(TAG, "Both sensors configured! Streaming live data:\n");

    uint8_t mpu_buf[6];
    uint8_t ppg_buf[6];

    while (1) {
        // 1. Read MPU6500 Accelerometer (6 bytes: X, Y, Z)
        if (i2c_read_reg(MPU6500_ADDR, MPU_REG_ACCEL_XOUT_H, mpu_buf, 6) == ESP_OK) {
            int16_t ax = (int16_t)((mpu_buf[0] << 8) | mpu_buf[1]);
            int16_t ay = (int16_t)((mpu_buf[2] << 8) | mpu_buf[3]);
            int16_t az = (int16_t)((mpu_buf[4] << 8) | mpu_buf[5]);

            // 2. Read MAX30102 FIFO sample (3 bytes Red, 3 bytes IR)
            if (i2c_read_reg(MAX30102_ADDR, MAX_REG_FIFO_DATA, ppg_buf, 6) == ESP_OK) {
                uint32_t red = ((uint32_t)(ppg_buf[0] & 0x03) << 16) | ((uint32_t)ppg_buf[1] << 8) | ppg_buf[2];
                uint32_t ir  = ((uint32_t)(ppg_buf[3] & 0x03) << 16) | ((uint32_t)ppg_buf[4] << 8) | ppg_buf[5];

                printf("IMU [Ax:%6d Ay:%6d Az:%6d] | PPG [Red:%6lu IR:%6lu]\n", ax, ay, az, red, ir);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(100)); // Stream at 10 Hz
    }
}