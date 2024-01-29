#include <filesystem>
#include <string>
#include <set>
#include <vector>
#include <glm/glm.hpp>

class LfLoader
{
    public:
    using DataGrid = std::vector<std::vector<std::vector<float>>>;
    glm::ivec2 getColsRows() const {return colsRows;}
    void loadData(std::string path);
    size_t imageSize() const {return resolution.x*resolution.y*resolution.z;}
    glm::ivec3 imageResolution() const {return resolution;} 
    size_t imageCount() const {return colsRows.x*colsRows.y;} 
    std::vector<float> image(glm::ivec2 colRow) {return grid[colRow.x][colRow.y];} 
    static void saveOutput(const float* rgb, int width, int height, std::string filename);

    private:
    glm::ivec3 resolution{0};
    glm::ivec2 colsRows{0};
    DataGrid grid;
    void initGrid(glm::uvec2 inColsRows);
    const std::set<std::filesystem::path> listPath(std::string path) const;
    glm::ivec2 parseFilename(std::string name) const;
    void loadImage(std::string path, glm::uvec2 coords);
};
