//
//  view_delegate.hpp
//  escg
//
//  Created by Louie Sinadjan on 06/01/2025.
//

#pragma once
#include "config.h"
#include "../view/renderer.hpp"
class ViewDelegate : public MTK::ViewDelegate
{
    public:
        ViewDelegate(MTL::Device* device);
        virtual ~ViewDelegate() override;
        virtual void drawInMTKView(MTK::View* view) override;

    private:
        Renderer* renderer;
};
