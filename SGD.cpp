#include <iostream>
#include <cmath>
#include <ctime>
#include <limits>
#include <string>
#include <array>
#include <vector>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <algorithm>
#include <random>
#include <chrono>

struct Point {
    float x, y;
};

#define MAX_iteration 50
#define EPS 0.001
#define delta 1.0e-6
#define learning_rate 1.0

int count = 0;
float distanceSquared = 0.0f;
float Error = std::numeric_limits<float>::max();
float initialError;
bool initialized = false;
float preError = 0.0f;
float dError = std::numeric_limits<float>::max();
float errorHistory[MAX_iteration];

float random_x;
float random_y;
float angle = 0.0f;

std::array<float, 3> motion = {0.0f, 0.0f, 0.0f};

size_t m_batch_size = 22;

std::vector<Point> read_scan_points(const std::string& file_path) {
    std::ifstream file(file_path);
    std::vector<Point> points;
    if (!file.is_open()) {
        std::cerr << "File could not be opened." << std::endl;
        return points;
    }
    std::string line_str;
    while (std::getline(file, line_str)) {
        std::istringstream iss(line_str);
        float x, y;
        if (!(iss >> x >> y)) {
            std::cerr << "Failed to parse line: " << line_str << std::endl;
            continue;
        }
        points.push_back({x, y});
    }
    return points;
}

Point calculate_average(const std::vector<Point>& points) {
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (const auto& point : points) {
        sum_x += point.x;
        sum_y += point.y;
    }
    float avg_x = sum_x / points.size();
    float avg_y = sum_y / points.size();
    return {avg_x, avg_y};
}

std::array<std::array<float, 3>, 3> make_transformation_matrix(float tx, float ty, float theta) {
    return {{
        {std::cos(theta), -std::sin(theta), tx},
        {std::sin(theta), std::cos(theta), ty},
        {0.0f, 0.0f, 1.0f}
    }};
}

std::vector<Point> transformpoints(const std::vector<Point>& points, float dx, float dy, double theta) {
    std::vector<Point> moved_points;
    auto transformation_matrix = make_transformation_matrix(dx, dy, theta);
    for (const auto& point : points) {
        float new_x = transformation_matrix[0][0] * point.x + transformation_matrix[0][1] * point.y + transformation_matrix[0][2];
        float new_y = transformation_matrix[1][0] * point.x + transformation_matrix[1][1] * point.y + transformation_matrix[1][2];
        moved_points.push_back({new_x, new_y});
    }
    return moved_points;
}

void plot(FILE* gnuplot_pipe, const std::vector<Point>& target, const std::vector<Point>& Source, int iteration) {
    fprintf(gnuplot_pipe, "set size ratio 1\n");
    fprintf(gnuplot_pipe, "set xrange [-20:20]\n");
    fprintf(gnuplot_pipe, "set yrange [-20:20]\n");
    fprintf(gnuplot_pipe, "set title 'Iteration %d'\n", iteration);
    fprintf(gnuplot_pipe, "plot '-' with points pointtype 7 pointsize 1 lc rgb 'blue' title 'Target points','-' with points pointtype 7 pointsize 1 lc rgb 'red' title 'Source points'\n");
    for (const auto& point : target) {
        fprintf(gnuplot_pipe, "%f %f\n", point.x, point.y);
    }
    fprintf(gnuplot_pipe, "e\n");
    for (const auto& point : Source) {
        fprintf(gnuplot_pipe, "%f %f\n", point.x, point.y);
    }
    fprintf(gnuplot_pipe, "e\n");
    fflush(gnuplot_pipe);
}

float distance(const Point& points, const Point& point) {
    return sqrt((points.x - point.x) * (points.x - point.x) + (points.y - point.y) * (points.y - point.y));
}

int findClosestPoint(const Point& point, const std::vector<Point>& target) {
    int Index = -1;
    float minDist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < target.size(); ++i) {
        float dist = distance(target[i], point);
        if (dist < minDist) {
            minDist = dist;
            Index = i;
        }
    }
    return Index;
}

float diffx(Point Target, Point SOurce) {
    float fx_delta = (Target.x - (SOurce.x + delta)) * (Target.x - (SOurce.x + delta)) + (Target.y - SOurce.y) * (Target.y - SOurce.y);
    float fx = (Target.x - SOurce.x) * (Target.x - SOurce.x) + (Target.y - SOurce.y) * (Target.y - SOurce.y);
    return (fx_delta - fx) / delta;
}

float diffy(Point Target, Point SOurce) {
    float fx_delta = (Target.x - SOurce.x) * (Target.x - SOurce.x) + (Target.y - (SOurce.y + delta)) * (Target.y - (SOurce.y + delta));
    float fx = (Target.x - SOurce.x) * (Target.x - SOurce.x) + (Target.y - SOurce.y) * (Target.y - SOurce.y);
    return (fx_delta - fx) / delta;
}

float difftheta(Point Target, Point SOurce) {
    float fx_delta = (Target.x - ((SOurce.x) * cos(delta * M_PI / 180) - (SOurce.y) * sin(delta * M_PI / 180))) * (Target.x - ((SOurce.x) * cos(delta * M_PI / 180) - (SOurce.y) * sin(delta * M_PI / 180))) + (Target.y - ((SOurce.x) * (sin(delta * M_PI / 180)) + (SOurce.y) * cos(delta * M_PI / 180))) * (Target.y - ((SOurce.x) * (sin(delta * M_PI / 180)) + (SOurce.y) * cos(delta * M_PI / 180)));
    float fx = (Target.x - SOurce.x) * (Target.x - SOurce.x) + (Target.y - SOurce.y) * (Target.y - SOurce.y);
    return (fx_delta - fx) / delta;
}

void shuffle_data(std::vector<Point>& points) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(points.begin(), points.end(), g);
}

std::vector<Point> createBatch(std::vector<Point>& source, size_t& m_current_offset, size_t m_batch_size) {
    std::vector<Point> batch;
    auto target_offset = m_current_offset + m_batch_size;

    while (target_offset >= source.size()) {
        while (m_current_offset < source.size()) {
            batch.push_back(source[m_current_offset++]);
        }
        shuffle_data(source);
        m_current_offset = 0;
        target_offset = target_offset - source.size();
    }
    while (m_current_offset < target_offset) {
        batch.push_back(source[m_current_offset++]);
    }

    return batch;
}

std::vector<Point>  icp_scan_matching(FILE* gnuplot_pipe, std::vector<Point>& Source, const std::vector<Point>& target) {
    float previous_error_sum = std::numeric_limits<float>::max();
    size_t m_current_offset = 0;
    shuffle_data(Source);

    for (int iter = 0; iter <= MAX_iteration; ++iter) {
        auto batch = createBatch(Source, m_current_offset, m_batch_size);
        //  std::cout << "Iteration " << iter << ", Batch size: " << batch.size() << std::endl;
          std::cout << "Iteration " << iter << ", offset: " << m_current_offset << std::endl;

        std::vector<Point> target_closest;
        float error_sum = 0;
        double gradDx = 0;
        double gradDy = 0;
        double gradTheta = 0;
        float dx = 0;
        float dy = 0;
        float dth = 0;

        for (const auto& source : batch) {
            int index = findClosestPoint(source, target);
            target_closest.push_back(target[index]);
            Point error = {target[index].x - source.x, target[index].y - source.y};
            Point Target = {target[index].x, target[index].y};
            error_sum += error.x * error.x + error.y * error.y;
            gradDx += diffx(Target, source);
            gradDy += diffy(Target, source);
            gradTheta += difftheta(Target, source);
        }

        int num_points = batch.size();

        dx = (-gradDx / num_points) * learning_rate;
        dy = (-gradDy / num_points) * learning_rate;
        dth = (-gradTheta / num_points) * learning_rate;

        //td::cout << "dth: " << dth << std::endl;

        for (auto& source : Source) {
            float x_new = source.x * cos(dth) - source.y * sin(dth);
            float y_new = source.x * sin(dth) + source.y * cos(dth);
            source.x = x_new + dx;
            source.y = y_new + dy;
        }

        //plot(gnuplot_pipe, target, Source, iter);

        if (std::abs(previous_error_sum - error_sum) < EPS) {
            std::cout << "Converged after " << iter << " iterations." << std::endl;
            break;
        }
        previous_error_sum = error_sum;
    }
    return Source;
}

int main(void) {
    std::vector<Point> current = read_scan_points("scan_1.txt");
    std::vector<Point> target = read_scan_points("scan_2.txt");

    std::cout << "Points from scan_1.txt:" << std::endl;
    for (const auto& point : current) {
        std::cout << "x: " << point.x << ", y: " << point.y << std::endl;
    }

    std::cout << "Points from scan_2.txt:" << std::endl;
    for (const auto& point : target) {
        std::cout << "x: " << point.x << ", y: " << point.y << std::endl;
    }

    Point avg1 = calculate_average(current);
    std::cout << "Average of points in scan_1.txt: x: " << avg1.x << ", y: " << avg1.y << std::endl;

    float dx = 1.0f;
    float dy = 0.25f;
    double theta = -M_PI / 8;
    std::vector<Point> moved_current = transformpoints(current, dx, dy, theta);

    std::cout << "Moved points from scan_1.txt:" << std::endl;
    for (const auto& point : moved_current) {
        std::cout << "x: " << point.x << ", y: " << point.y << std::endl;
    }
    std::vector<Point> Source = moved_current;

    FILE* gnuplot_pipe = popen("gnuplot -persistent", "w");
    if (!gnuplot_pipe) {
        std::cerr << "Could not open pipe to GNUplot." << std::endl;
        return 1;
    }
    plot(gnuplot_pipe, target, Source, 0);

    int second = 5;
    printf("%d秒間止まります。\n", second);
    sleep(second);

    // shuffle_data(Source);
    // for (size_t i = 0; i < Source.size(); ++i) {
    //     std::cout << i << ": x: " << Source[i].x << ", y: " << Source[i].y << std::endl;
    // }
auto start_time = std::chrono::high_resolution_clock::now();
    //icp_scan_matching(gnuplot_pipe, Source, target);
   std::vector<Point> final_transformed_source = icp_scan_matching(gnuplot_pipe, Source, target);
    auto end_time = std::chrono::high_resolution_clock::now();
   
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "ICP algorithm completed in " << duration.count() << " millseconds." << std::endl;
    plot(gnuplot_pipe, target, final_transformed_source, MAX_iteration);

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // std::cout << "ICP algorithm completed in " << duration.count() << " milliseconds." << std::endl;
 pclose(gnuplot_pipe);
    return 0;
}