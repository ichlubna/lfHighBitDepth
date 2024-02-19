#include "interpolator.h"
#include "libs/arguments/arguments.hpp"

int main(int argc, char **argv)
{
    Arguments args(argc, argv);
    std::string helpText{ "Usage:\n"
                          "-i - folder with lf grid images\n"
                          "     this application expects four input floating point RGB images in EXR format, their coordinates will be set according to the order sorted by their names\n"
                          "     when sorted the expected camera positions are 1. top left, 2. top right, 3. bottom left, 4. bottom right\n"
                          "-c - camera position in normalized coordinates of the grid in format: x_y, default: 0.5_0.5 (creates view exactly between the four views)\n"
                          "-o - output path\n"
                          "-r - normalized scanning range - the maximum and minimum disparity between input images, default is from zero to half of image width - 0.0_0.5\n"
                          "-g - aspect ratio of the capturing camera spacing in horizontal / vertical axis, 1 means that the cameras are spaced equally in x and y\n"
                          "     if this option is not used or set to 0, the lf grid spacing is expected to be in the same ratio as resolution \n"
                          "-t - the number of kernel runs for time performance measurements\n"
                          "-p - float parameter for debug purposes\n"
                          "Example:\n"
                          "lfHighBitDepth -i /MyAmazingMachine/theImages -o ./outputs -c 0.5_0.5 -g 1 -r 0.0_0.1\n"
                        };
    if(args.printHelpIfPresent(helpText))
        return 0;
    
    if(!args["-i"] || !args["-o"])
    {
        std::cerr << "Missing required parameters. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }
    
    try
    {
        std::string coords{"0.5_0.5"};
        if(args["-c"])
            coords = static_cast<std::string>(args["-c"]);
        std::string range{"0.0_0.5"};
        if(args["-r"])
            range = static_cast<std::string>(args["-r"]);

        Interpolator interpolator(static_cast<std::string>(args["-i"]), args["-g"]);
        Interpolator::InterpolationParams params;
        params
        .setCoordinates(coords)
        ->setScanRange(range)
        ->setOutputPath(static_cast<std::string>(args["-o"]))
        ->setParameter(static_cast<float>(args["-p"]))
        ->setRuns(std::stoi(static_cast<std::string>(args["-t"])));
        interpolator.interpolate(params);
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
