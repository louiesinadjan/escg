//
//  app_delegate.hpp
//  escg
//
//  Created by Louie Sinadjan on 06/01/2025.
//

#pragma once
#include "../config.hpp"
#include "view_delegate.hpp"

class AppDelegate : public NS::ApplicationDelegate
{
    public:
        ~AppDelegate();

        virtual void applicationWillFinishLaunching(NS::Notification* notification) override;
        virtual void applicationDidFinishLaunching(NS::Notification* notification) override;
        virtual bool applicationShouldTerminateAfterLastWindowClosed(NS::Application* sender) override;

    private:
        NS::Window* window;
        MTK::View* mtkView;
        MTL::Device* device;
        ViewDelegate* viewDelegate = nullptr;
};
