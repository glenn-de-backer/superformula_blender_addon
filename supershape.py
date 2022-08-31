bl_info = {
    "name": "SuperFormula",
    "blender": (2, 80, 0),
    "category": "Mesh",
}

import bpy


import numpy as np
import bpy
import bmesh
from mathutils import Vector


def supercoords(params, shape=(50,50)):
    '''Returns coordinates of a parametrized 3D supershape.
    
    See
    http://paulbourke.net/geometry/supershape/
    https://en.wikipedia.org/wiki/Superformula

    Params
    ------
    params: 1x6 or 2x6 array
        Parameters of the two supershapes. If 1x6 the same
        parameters will be used for the second supershape.
        The order of the parameters is as follows:
        m: float
            long/lat frequency.
            Defaults to 0.01
        a: float
            long/lat inverse amplitude of first term.
            Defaults to 1.
        b: float
            long/lat inverse amplitude of second term
            Defaults to 1.
        n1: float
            First exponent. Defaults to 0.1
        n2: float
            Second exponent. Defaults to 0.01
        n3: float
            Third exponent. Actually (-1/n3). Defaults to 10.0
    shape : tuple
        longitude/latitude resolution (U,V)

    Returns
    -------
    x: UxV array
        x coordinates for each long/lat point
    y: UxV array
        y coordinates for each long/lat point
    z: UxV array
        z coordinates for each long/lat point
    '''

    params = np.atleast_2d(params)
    if params.shape[0] == 1:
        params = np.tile(params, (2,1))

    sf = lambda alpha, sp: (
        np.abs(np.cos(sp[0]*alpha/4.)/sp[1])**sp[4] + 
        np.abs(np.sin(sp[0]*alpha/4.)/sp[2])**sp[5]
    )**(-1/sp[3])

    u = np.linspace(-np.pi, np.pi, shape[0]) # long., theta
    v = np.linspace(-np.pi/2, np.pi/2, shape[1]) # lat., phi
        
    g = np.meshgrid(v, u)
    uv = np.stack((g[1],g[0]),-1)
    r1 = sf(uv[...,0], params[0])
    r2 = sf(uv[...,1], params[1])    

    x = r1 * np.cos(u)[:,None] * r2 * np.cos(v)[None, :]
    y = r1 * np.sin(u)[:,None] * r2 * np.cos(v)[None, :]
    z = r2 * np.sin(v)[None, :]

    return x,y,z

def make_bpy_mesh(shape, name='supershape', coll=None, smooth=True, weld=False):
    '''Create a Blender (>2.8) mesh from supershape coordinates.
    Adapted from
    http://wiki.theprovingground.org/blender-py-supershape
    Params
    ------
    shape : tuple
        long./lat. resolution of supershape
    Returns
    -------
    obj: bpy.types.Object
        Mesh object build from quads.
    name: str
        Name of object.
    coll: bpy collection
        Collection to link object to. If None,
        default collection is used. If False, object is not
        added to any collection.
    smooth: bool
        Smooth or flat rendering.
    weld: bool, optional
        Whether to add a weld-modifier to the mesh. The weld modifier
        closes geometry seams by merging duplicate vertices. Defaults to
        false.
    '''
    U, V = shape
    xy = np.stack(np.meshgrid(np.linspace(0, 1, V),
                              np.linspace(0, 1, U)), -1).astype(np.float32)
    vertices = np.concatenate(
        (xy, np.zeros((U, V, 1), dtype=np.float32)), -1).reshape(-1, 3)

    # Vertices
    bm = bmesh.new()
    for v in vertices:
        bm.verts.new(Vector(v))
    # Required after adding / removing vertices and before accessing them by index.
    bm.verts.ensure_lookup_table()
    # Required to actually retrieve the indices later on (or they stay -1).
    bm.verts.index_update()
    # Faces
    for u in range(U-1):
        for v in range(V-1):
            A = u*V + v
            B = u*V + (v+1)
            C = (u+1)*V + (v+1)
            D = (u+1)*V + v
            bm.faces.new((bm.verts[D], bm.verts[C],
                          bm.verts[B], bm.verts[A]))
    # UV
    uv_layer = bm.loops.layers.uv.new()
    for face in bm.faces:
        for loop in face.loops:
            v, u = vertices[loop.vert.index][:2]
            loop[uv_layer].uv = (u, 1.-v)

    bm.normal_update()
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    if smooth:
        for f in mesh.polygons:
            f.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)
    del mesh

    if weld:
        mod = obj.modifiers.new("CloseSeams", 'WELD')
        mod.merge_threshold = 1e-3
    if coll is None:
        coll = bpy.context.collection
    if coll is not False:
        coll.objects.link(obj)

    return obj

def update_bpy_mesh(x, y, z, obj):
    '''Update a Blender (>2.8) mesh from supershape coordinates.
    Adapted from
    http://wiki.theprovingground.org/blender-py-supershape
    Params
    ------
    x: UxV array
        x coordinates for each long/lat point
    y: UxV array
        y coordinates for each long/lat point
    z: UxV array
        z coordinates for each long/lat point
    obj: bpy.types.Object
        Object to update. Note that the long./lat. resolution must match.
    '''
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    flat = np.concatenate((x, y, z), -1)
    obj.data.vertices.foreach_set("co", flat.reshape(-1))

    # Update normals
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # Instead of closing seams at data level through
    # bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-3)
    # use a weld mesh modifier.
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.clear()
    obj.data.update()
    bm.free()
    del bm

class ObjectSuperFormula(bpy.types.Operator):
    """My Object Moving Script"""      # Use this as a tooltip for menu items and buttons.
    bl_idname = "mesh.superformula"        # Unique identifier for buttons and menu items to reference.
    bl_label = "SuperFormula mesh"         # Display name in the interface.
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.
    
    smooth: bpy.props.BoolProperty(name="Smooth", default=True)
    m: bpy.props.FloatProperty(name="M1", default=7.0, step=50)
    a: bpy.props.FloatProperty(name="A1", default=1.0, step=1)
    b: bpy.props.FloatProperty(name="B1", default=1.0, step=1)
    sync: bpy.props.BoolProperty(name="Sync", default=True)
    m2: bpy.props.FloatProperty(name="M2", default=7.0, step=50)
    a2: bpy.props.FloatProperty(name="A2", default=1.0, step=1)
    b2: bpy.props.FloatProperty(name="B2", default=1.0, step=1)

    def execute(self, context):        # execute() is called when running the operator.
        # Generate supershape
        shape = (100, 100)
        SHAPE_1 = [self.m, self.a, self.b, 0.2, 1.7, 1.7]
        SHAPE_2 = [self.m2, self.a2, self.b2, 0.2, 1.7, 1.7]
                
        obj = make_bpy_mesh(shape, smooth=self.smooth, weld=True)
        x, y, z = supercoords([SHAPE_1, SHAPE_2], shape=shape)
        update_bpy_mesh(x, y, z, obj)

        return {'FINISHED'}            # Lets Blender know the operator finished successfully.

def menu_func(self, context):
    self.layout.operator(ObjectSuperFormula.bl_idname)

def register():
    bpy.utils.register_class(ObjectSuperFormula)
    bpy.types.VIEW3D_MT_object.append(menu_func)  # Adds the new operator to an existing menu.

def unregister():
    bpy.utils.unregister_class(ObjectSuperFormula)


# This allows you to run the script directly from Blender's Text editor
# to test the add-on without having to install it.
if __name__ == "__main__":
    register()