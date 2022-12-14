import numpy as np
import bpy
import bmesh
from mathutils import Vector

bl_info = {
    "name": "SuperFormula Addon",
    "author": "Glenn De Backer",
    "description": "An addond that makes it possible to generate meshes based on the superformula logic",
    "version": (1,0,1),
    "blender": (3, 0, 0),
    "location": "View3D > Add > Mesh",
    "doc_url": "https://github.com/glenn-de-backer/superformula_blender_addon",
    "support": "COMMUNITY",
    "category": "Add Mesh",
}


def supercoords(params, shape=(50, 50)):
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
        params = np.tile(params, (2, 1))

    def sf(alpha, sp): return (
        np.abs(np.cos(sp[0]*alpha/4.)/sp[1])**sp[4] +
        np.abs(np.sin(sp[0]*alpha/4.)/sp[2])**sp[5]
    )**(-1/sp[3])

    u = np.linspace(-np.pi, np.pi, shape[0])  # long., theta
    v = np.linspace(-np.pi/2, np.pi/2, shape[1])  # lat., phi

    g = np.meshgrid(v, u)
    uv = np.stack((g[1], g[0]), -1)
    r1 = sf(uv[..., 0], params[0])
    r2 = sf(uv[..., 1], params[1])

    x = r1 * np.cos(u)[:, None] * r2 * np.cos(v)[None, :]
    y = r1 * np.sin(u)[:, None] * r2 * np.cos(v)[None, :]
    z = r2 * np.sin(v)[None, :]

    return x, y, z


def make_bpy_mesh(shape, name='supershape', coll=None, smooth=True, weld=False, subdivide=False):
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
    if subdivide:
        mod_subsurf = obj.modifiers.new("Subdivide mesh", 'SUBSURF')
        mod_subsurf.levels = 1
        mod_subsurf.render_levels = 1            
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


class ObjectSuperFormula3D(bpy.types.Operator):
    # Definition
    """Superformale 3D Mesh"""
    bl_idname = "mesh.superformula_3d"
    bl_label = "SuperFormula 3D Mesh"
    bl_options = {'REGISTER', 'UNDO'}  # Enable undo for the operator.

    #
    #  Properties
    smooth: bpy.props.BoolProperty(
        name="Smooth",
        description="Enable smooth shading",
        default=True
    )

    weld: bpy.props.BoolProperty(
        name="Weld",
        description="Add weld operator",
        default=True
    )    

    subdivide: bpy.props.BoolProperty(
        name="Subdivide",
        description="Add subdivision",
        default=False
    )    

    # Resolution
    resolution_long: bpy.props.IntProperty(
        name="Resolution long",
        default=100
    )

    resolution_lat: bpy.props.IntProperty(
        name="Resolution lat",
        default=100
    )

    # Shape 1
    m: bpy.props.FloatProperty(
        name="M1",
        default=7.0,
        step=1
    )

    a: bpy.props.FloatProperty(
        name="A1",
        default=1.0,
        step=1
    )

    b: bpy.props.FloatProperty(
        name="B1",
        default=1.0,
        step=1
    )

    n1: bpy.props.FloatProperty(
        name="N1",
        default=0.2,
        step=1
    )
    n2: bpy.props.FloatProperty(
        name="N2",
        default=1.7,
        step=1
    )

    n3: bpy.props.FloatProperty(
        name="N3",
        default=1.7,
        step=1
    )

    sync: bpy.props.BoolProperty(
        name="Use same parameters as Shape 1",
        default=False
    )

    # Shape 2
    m2: bpy.props.FloatProperty(
        name="M2",
        default=7.0,
        step=50
    )

    a2: bpy.props.FloatProperty(
        name="A2",
        default=1.0,
        step=1
    )

    b2: bpy.props.FloatProperty(
        name="B2",
        default=1.0,
        step=1
    )

    n1_2: bpy.props.FloatProperty(
        name="N1_2",
        default=0.2,
        step=1
    )

    n2_2: bpy.props.FloatProperty(
        name="N2_2",
        default=1.7,
        step=1
    )

    n3_2: bpy.props.FloatProperty(
        name="N3_2",
        default=1.7,
        step=1
    )

    scale: bpy.props.FloatVectorProperty(
        name="Scale",
        default=(1.0, 1.0, 1.0),
        description="Scale object",
        subtype="XYZ"
    )

    def shapes_update(self, context):
        self.sync = False                
        shape_values = []
        match self.shapes:
            case "Default" :
                self.weld = True
                shape_values = [
                    [7.0, 1.0, 1.0, 0.2, 1.7, 1.7],
                    [7.0, 1.0, 1.0, 0.2, 1.7, 1.7]    
                ]
            case "Starfish" :
                self.weld = True
                shape_values = [
                    [7.0, 1.0, 1.0, 0.2, 1.48, 1.48],
                    [1.95, 1.0, 1.0, 0.2, 1.12, 1.01]    
                ]
            case "Clover" :
                self.weld = False
                shape_values = [
                    [7.93, 1.0, 1.0, 0.10, 6.35, -0.23],
                    [4.0, -0.05, -0.05, 1.0, -0.28, 1.0]    
                ]      
            case "SharkTooth" :
                self.weld = False
                shape_values = [
                    [2.63, 1.03, 1.05, 0.29, 1.48, 1.48],
                    [-1.90, 1.31, 1.78, 0.20, 0.64, 0.95]    
                ]                              

        # update values
        self.m  = shape_values[0][0]
        self.a  = shape_values[0][1]
        self.b  = shape_values[0][2]
        self.n1 = shape_values[0][3]
        self.n2 = shape_values[0][4]
        self.n3 = shape_values[0][5]

        self.m2   = shape_values[1][0]
        self.a2   = shape_values[1][1]
        self.b2   = shape_values[1][2]
        self.n1_2 = shape_values[1][3]
        self.n2_2 = shape_values[1][4]
        self.n3_2 = shape_values[1][5]


    shapes: bpy.props.EnumProperty(
        name="Examples",
        description="Examples of meshes",
        items=[
            ('Default', "Default", ""),
            ('Starfish', "Starfish", ""),
            ('Clover', "Clover", ""),
            ('SharkTooth', "Shark Tooth", ""),            
        ],
        update=shapes_update
    )

    def draw(self, context):
        layout = self.layout

        row = layout.row(align=True)
        row.prop(self, "smooth")
        row.prop(self, "weld")

        # shapes
        row = layout.row()
        row.prop(self, "shapes")

        # box resolution
        boxResolution = layout.box()
        boxResolution.label(text="Resolution")
        boxResolution.prop(self, "resolution_long")
        boxResolution.prop(self, "resolution_lat")

        # box shape 1
        boxShape1 = layout.box()
        boxShape1.label(text="Shape 1 definition")
        boxShape1.prop(self, "m")
        boxShape1.prop(self, "a")
        boxShape1.prop(self, "b")
        boxShape1.prop(self, "n1")
        boxShape1.prop(self, "n2")
        boxShape1.prop(self, "n3")

        # box shape 1
        boxShape2 = layout.box()
        boxShape2.label(text="Shape 2 definition")
        boxShape2.prop(self, "sync")
        boxShape2.prop(self, "m2")
        boxShape2.prop(self, "a2")
        boxShape2.prop(self, "b2")
        boxShape2.prop(self, "n1_2")
        boxShape2.prop(self, "n2_2")
        boxShape2.prop(self, "n3_2")

        # Scale
        boxScale = layout.box()
        boxScale.prop(self, "scale")
        boxScale.prop(self, "subdivide")

    # Execute operator
    def execute(self, context):

        # define shape resolution
        shape = (self.resolution_long, self.resolution_lat)

        # create shape 1 and shape 2
        SHAPE_1 = [self.m, self.a, self.b, self.n1, self.n2, self.n3]
        SHAPE_2 = []

        # check if sync is enabled or not
        if self.sync == False:
            SHAPE_2 = [self.m2, self.a2, self.b2,
                       self.n1_2,  self.n2_2,  self.n3_2]
        else:
            SHAPE_2 = [self.m, self.a, self.b, self.n1, self.n2, self.n3]

            # update properties
            self.m2 = self.m
            self.a2 = self.a
            self.b2 = self.b
            self.n1_2 = self.n1
            self.n2_2 = self.n2
            self.n3_2 = self.n3

        # create object
        obj = make_bpy_mesh(shape, smooth=self.smooth, weld=self.weld, subdivide=self.subdivide)

        # generate shape
        x, y, z = supercoords([SHAPE_1, SHAPE_2], shape=shape)

        # update mesh
        update_bpy_mesh(x, y, z, obj)

        # scale mesh
        obj.scale = self.scale

        # return that it's finished
        return {'FINISHED'}


# register
def add_object_button(self, context):
    # Add mesh menu item
    self.layout.operator(ObjectSuperFormula3D.bl_idname,
                         text="Add SuperFormula 3D Mesh", icon='PLUGIN')


def menu_func(self, context):
    # Search menu function
    self.layout.operator(ObjectSuperFormula3D.bl_idname)


def register():
    # Register operator
    bpy.utils.register_class(ObjectSuperFormula3D)
    bpy.types.VIEW3D_MT_object.append(menu_func)
    bpy.types.VIEW3D_MT_mesh_add.append(add_object_button)


def unregister():
    # Unregister operator
    bpy.utils.unregister_class(ObjectSuperFormula3D)
    bpy.types.VIEW3D_MT_mesh_add.remove(add_object_button)


if __name__ == "__main__":
    register()
