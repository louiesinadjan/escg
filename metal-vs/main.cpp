#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "config.hpp"
#include "control/app_delegate.hpp"

int main(int argc, char *argv[]) {
    NS::AutoreleasePool *autoreleasePool = NS::AutoreleasePool::alloc()->init();

    AppDelegate controller;

    NS::Application *app = NS::Application::sharedApplication();
    app->setDelegate(&controller);
    app->run();

    autoreleasePool->release();

    printf("return");
    
    return 0;
}
