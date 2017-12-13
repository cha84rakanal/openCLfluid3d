//
//  main.cpp
//  fluid3d_cl
//
//  Created by Tatsuya Minagawa on 2017/12/13.
//  Copyright © 2017年 Tatsuya Minagawa. All rights reserved.
//

#include "utils.h"
#include "rx_trackball.h"

#include <iostream>
#include <GLUT/glut.h>

#include "CLsolver.cpp"



/* macros */

#define IX3(i,j,k) ((i)+(f3d.N+2)*(j)+(f3d.N+2)*(f3d.N+2)*k)

/* global variables */

int dvel;
int win_id;
int win_x, win_y;
int mouse_down[3];
int omx, omy, mx, my;

// ウィンドウ情報
int g_iWinW = 512;        //!< 描画ウィンドウの幅
int g_iWinH = 512;        //!< 描画ウィンドウの高さ

CLFluid3d f3d;

/*
 ----------------------------------------------------------------------
 OpenGL specific drawing routines
 ----------------------------------------------------------------------
 */

void pre_display ( void ){
    glViewport ( 0, 0, win_x, win_y );
    glMatrixMode ( GL_PROJECTION );
    glLoadIdentity ();
    gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );
    //gluPerspective(45.0f, (float)g_iWinW/(float)g_iWinH, 0.2f, 1000.0f);
    glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
    glClear ( GL_COLOR_BUFFER_BIT );
}

/*
 ----------------------------------------------------------------------
 relates mouse movements to forces sources
 ----------------------------------------------------------------------
 */

void get_from_UI3 ( float * d, float * u, float * v ,float *w){
    int i, j, size = (f3d.N+2)*(f3d.N+2)*(f3d.N+2);
    
    for ( i=0 ; i<size ; i++ ) {
        u[i] = v[i] = w[i] = d[i] = 0.0f;
    }
    
    //u[IX3(26,39,30)] = f3d.force;
    v[IX3(26,39,30)] = f3d.force;
    d[IX3(26,39,30)] = f3d.source;
    
    //printf("aaaa\n");
    
    /*
     if ( !mouse_down[0] && !mouse_down[2] ) return;
     
     i = (int)((       mx /(float)win_x)*f3d.N+1);
     j = (int)(((win_y-my)/(float)win_y)*f3d.N+1);
     
     if ( i<1 || i>f3d.N || j<1 || j>f3d.N ) return;
     
     if ( mouse_down[0] ) {
     u[IX(i,j)] = f3d.force * (mx-omx);
     v[IX(i,j)] = f3d.force * (omy-my);
     }
     
     if ( mouse_down[0] ) {
     d[IX(i,j)] = f3d.source;
     std::cout << i << "," << j << std::endl;
     }
     
     */
    
    omx = mx;
    omy = my;
    
    return;
}

/*
 ----------------------------------------------------------------------
 GLUT callback routines
 ----------------------------------------------------------------------
 */

static void key_func ( unsigned char key, int x, int y ){
    switch ( key ){
        case 'c':
        case 'C':
            f3d.clear_data ();
            break;
            
        case 'q':
        case 'Q':
            f3d.free_data ();
            exit ( 0 );
            break;
            
        case 'v':
        case 'V':
            dvel = !dvel;
            break;
    }
}


static void sp_key_func(int key, int x, int y){
    switch(key){
        case GLUT_KEY_UP:
            f3d.layer++;
            std::cout << f3d.layer << endl;
            break;
        case GLUT_KEY_DOWN:
            f3d.layer--;
            std::cout << f3d.layer << endl;
            break;
    }
    glutPostRedisplay();
}

void mouse_func ( int button, int state, int x, int y ){
    omx = mx = x;
    omx = my = y;
    
    mouse_down[button] = state == GLUT_DOWN;
}

void motion_func ( int x, int y ){
    mx = x;
    my = y;
}

void reshape_func ( int w, int h){
    //glutSetWindow ( win_id );
    //glutReshapeWindow ( width, height );
    
    //win_x = width;
    //win_y = height;
    
    g_iWinW = w;
    g_iWinH = h;
    
    glViewport(0, 0, w, h);
    //g_tbView.SetRegion(w, h);
    
    // 透視変換行列の設定
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)g_iWinW/(float)g_iWinH, 0.2f, 1000.0f);
    //gluLookAt(3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    //gluLookAt( 100.f, 150.0f, 150.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f );
    
    // モデルビュー変換行列の設定
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void idle_func ( void ){
    //get_from_UI3 ( f3d.dens_prev, f3d.u_prev, f3d.v_prev ,f3d.w_prev );
    //f3d.vel_step_call();
    //f3d.dens_step_call();
    f3d.cl_step();
    //glutSetWindow ( win_id );
    
    //display_funcの呼び出し
    glutPostRedisplay ();
    std::cout << "draw" << endl;
    
}

void display_func ( void ){
    pre_display ();
    
    if ( dvel ) f3d.draw_velocity_2d ();
    else        f3d.draw_density_2d ();
    
    glutSwapBuffers ();
}


/*
 ----------------------------------------------------------------------
 open_glut_window --- open a glut compatible window and set callbacks
 ----------------------------------------------------------------------
 */

void open_glut_window ( void ){
    glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE );
    
    glutInitWindowPosition ( 0, 0 );
    glutInitWindowSize ( win_x, win_y );
    win_id = glutCreateWindow ( "Alias | wavefront" );
    
    glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
    glClear ( GL_COLOR_BUFFER_BIT );
    glutSwapBuffers ();
    glClear ( GL_COLOR_BUFFER_BIT );
    glutSwapBuffers ();
    
    pre_display ();
    
    glutKeyboardFunc ( key_func );
    glutSpecialFunc(sp_key_func);
    glutMouseFunc ( mouse_func );
    glutMotionFunc ( motion_func );
    glutReshapeFunc ( reshape_func );
    glutIdleFunc ( idle_func );
    glutDisplayFunc ( display_func );
}


/*
 ----------------------------------------------------------------------
 main --- main routine
 ----------------------------------------------------------------------
 */

int main ( int argc, char ** argv ){
    glutInit ( &argc, argv );
    
    if ( argc != 1 && argc != 6 ) {
        fprintf ( stderr, "usage : %s N dt diff visc force source\n", argv[0] );
        fprintf ( stderr, "where:\n" );\
        fprintf ( stderr, "\t N      : grid resolution\n" );
        fprintf ( stderr, "\t dt     : time step\n" );
        fprintf ( stderr, "\t diff   : diffusion rate of the density\n" );
        fprintf ( stderr, "\t visc   : viscosity of the fluid\n" );
        fprintf ( stderr, "\t force  : scales the mouse movement that generate a force\n" );
        fprintf ( stderr, "\t source : amount of density that will be deposited\n" );
        exit ( 1 );
    }
    
    if ( argc == 1 ) {
        /*
         N = 64;
         dt = 0.1f;
         diff = 0.0f;
         visc = 0.0f;
         force = 5.0f;
         source = 100.0f;
         
         fprintf ( stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
         N, dt, diff, visc, force, source );
         */
    } else {
        /*
         N = atoi(argv[1]);
         dt = atof(argv[2]);
         diff = atof(argv[3]);
         visc = atof(argv[4]);
         force = atof(argv[5]);
         source = atof(argv[6]);
         */
    }
    
    printf ( "\n\nHow to use this demo:\n\n" );
    printf ( "\t Add densities with the right mouse button\n" );
    printf ( "\t Add velocities with the left mouse button and dragging the mouse\n" );
    printf ( "\t Toggle density/velocity display with the 'v' key\n" );
    printf ( "\t Clear the simulation by pressing the 'c' key\n" );
    printf ( "\t Quit by pressing the 'q' key\n" );
    
    dvel = 0;
    
    //init
    //f2d = new Fluid2d();
    fprintf ( stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
             f3d.N, f3d.dt, f3d.diff, f3d.visc, f3d.force, f3d.source );
    
    win_x = 512;
    win_y = 512;
    open_glut_window ();
    
    glutMainLoop ();
    
    exit ( 0 );
}

