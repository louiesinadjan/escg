//
//  renderer.hpp
//  escg
//
//  Created by Louie Sinadjan on 06/01/2025.
//
#pragma once
#include "../config.hpp"

class Renderer
{
    public:
        Renderer(MTL::Device* device);
        ~Renderer();
        void draw(MTK::View* view);

    private:
        MTL::Device* device;
        MTL::CommandQueue* commandQueue;
};
