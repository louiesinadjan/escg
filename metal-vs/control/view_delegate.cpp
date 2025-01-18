//
//  view_delegate.cpp
//  escg
//
//  Created by Louie Sinadjan on 06/01/2025.
//

#include "view_delegate.hpp"
ViewDelegate::ViewDelegate(MTL::Device *device)
    : MTK::ViewDelegate(), renderer(new Renderer(device)) {}

ViewDelegate::~ViewDelegate() { delete renderer; }

void ViewDelegate::drawInMTKView(MTK::View *view) { renderer->draw(view); }
