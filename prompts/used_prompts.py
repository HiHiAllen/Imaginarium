SCENE_HIERARCHICAL_TRAVERSAL_PROMPT = """Analyze the given image of an indoor or outdoor scene in a structured, hierarchical manner, adhering strictly to a predefined list of objects. Provide the results in a JSON format with the following steps:
1. **Identify ALL distinct areas or zones** in the scene, no matter how small or seemingly insignificant. Include transitional spaces, corners, and any visible partial areas.
2. **For EACH identified area, detect and list EVERY visible object**, focusing solely on parent object names and their associated child object names, **WITHOUT mentioning their locations or other relationships**. It is imperative that each object name is selected from a specified list of categories, referred to as predefined_objs_list: {predefined_eng_categories_list}. Ensure each object name is specific and clear. Consider the following from coarse to fine:
    a. Large elements (e.g., furniture, major appliances, architectural features) as parent objects
    b. Medium-sized objects (e.g., decorations, electronics, LCD_TV) as parent or child objects
    c. Small items (e.g., accessories, utensils, personal items) primarily as child objects
**Important Note**: Every identified object must be named according to the predefined_objs_list. This constraint ensures consistency and accuracy in the analysis. If an object does not fit any of the predefined objects, it should be carefully considered and matched with the closest available category, ensuring the description remains as accurate as possible.
**Ensure absolute thoroughness** in your analysis. No object, regardless of its size or perceived importance, should be omitted. Capture every detail visible in the image, from the largest architectural elements to the smallest discernible objects. If a parent object has no visible child objects, represent it with an empty array. If an object doesn't clearly belong to a parent object, list it as its own parent object with an empty array.
**Structure your response** as a comprehensive tree-structured JSON object with three levels of relationships: areas - parent objects - child objects. Each identified area should be a top-level key, with its value being an object containing parent objects as keys and arrays of their associated child objects as values.

**Example structure**:
```json
{{
  "area1": {{
    "parent_object1": ["child_object1", "child_object2"],
    "parent_object2": []
  }},
  "area2": {{
    "parent_object3": ["child_object3"]
  }}
}}
```
"""


SCENE_DEDUPLICATION_PROMPT = """You are an expert in object classification and natural language processing. Given the following list of object names from a scene description and the accompanying image, your task is to perform semantic deduplication. This involves identifying and removing redundant terms that refer to the same object, based on both the textual list and visual cues from the image. Preserve terms that describe distinct objects, even if they are of the same category.
List of objects:
{object_list}
Guidelines for deduplication:
1. Remove synonyms or alternative names for the same object (e.g., "framed artwork" and "framed artworks" likely refer to the same type of object).
2. Remove redundant terms that are more or less specific versions of the same object (e.g., "chandelier" and "ceiling light" if they likely refer to the same fixture).
3. Remove any non-specific terms or broad categories that do not refer to specific items, such as "living area," "furniture set," or general terms like "entertainment center" that encompass other specific items like "television.
4. Preserve terms that clearly indicate different objects, even if they are of the same category (e.g., "yellow pillow" and "blue pillow" should both be kept as they are distinct objects).
5. IMPORTANT: Always include the following tags in your final list, even if they were not in the original list: ['floor', 'wall', 'ceiling']. These are essential structural elements that should always be represented.
Finally, return only a python list of the objects.
"""

SECONDARY_INSPECTION_PROMPT= """You are an expert capable of accurately identifying all objects in images. Given an original image <image-placeholder> and a cropped portion of that image <image-placeholder>, please perceive the objects in the pictures from general to specific. I will also provide a List_Of_Detect_Items, which was used for visual grounding of the image content.
In the cropped image, areas marked in red are regions that the visual grounding model failed to identify. Please analyze each red-marked area one by one, numbering them from left to right, top to bottom. For each area, focus ONLY on the main object(s) that are predominantly within the red box. Ignore any objects that are only partially included and do not form a significant part of the marked area.
For each red-marked area, process the main object(s) according to the following three scenarios:
1. If there is a clear, complete object in the red-marked area that is not in the provided item list, please add this object to a new list.
2. If there is a clear, complete object in the red-marked area that already exists in the provided item list, you MUST give this object a more accurate name by adding concise attribute prefixes, such as "yellow book" or "crushed bottle". Then add this new, more specific name to the new list.
3. If the red-marked area is only a small part of an object already in the list, just ignore it.
Here is the List_Of_Detect_Items:
{detect_items_list_result}
Finally, please return ONLY a python list of newly added or renamed items in the format [obj1, obj2, ...]. This list should include both completely new objects and more specifically named versions of existing objects."""

SECONDARY_GET_ITEM_LIST_PROMPT= """Analyze the provided image <image-placeholder>, focusing on objects highlighted with red outlines. For each object, generate a specific and clear name, It is imperative that each object name is selected from a specified list of objects, referred to as predefined_objs_list: {predefined_eng_categories_list}.
Adhering to the following guidelines:
1. Specificity: Avoid vague terms like "debris" or "small items." Provide precise names for each object.
2. Semantic Grouping: Treat semantically related elements as a single object. For example, a "box covered with a tarp" should be identified as one object.
3. Comprehensive Identification: Ensure that all objects with red outlines are identified, capturing every detail visible in the image.
Return a python List.
"""


GENERATE_SCENE_GRAPH_PROMPT_S1= '''
Task Overview:
Create a scene graph for the objects identified in pic_1 <image-placeholder>, 
specifically those within the designated region, referred to as **items_in_region**: {items_in_region}.

Reference Images:
pic_2 <image-placeholder>: The complete scene image, listing all objects, is referred to as **all_items_list**: {all_items_list}.
{wall_color_name}

Object Attributes:
For each object in pic_1, populate the following attributes:

1.parent: Identify the parent object for the current item.
The parent is the object directly supporting the current item.
For example, 
If an item is placed on a table, the table is its parent. 
If an item is placed on the floor, the floor is its parent. 
If an item is attached to a wall, the wall is its parent.
If a book is stacked on a book, the book is its parent.

for each object, you should choose from the list:
{obj_near_str}

2.isAgainstWall: Determine if the object is directly against a wall. 
Determine if the object's back is directly against a wall, not the sides. If it isn't, set this to false; otherwise, set it to true.

3.directlyFacing: Determine if the object is directly facing another. Consider these scenarios:
A chair is directly facing a table if its front is oriented straight toward the table's center or a significant feature of the table.
A TV is directly facing a table if its screen is aligned with the table's center or key area.

Especially make sure the "parent" selection is correct.
Please ensure that the attribute selections are made from **all_items_list** and match the exact names, including the object label and index, such as ground_0.

Please follow these step:
step1:Identify the parent object for each object and give the reason.
step2:Identify the object's isAgainstWall attribute and give the reason.
step3:Identify the object's directlyFacing attribute and give the reason.
step4:Output the result in the following format:

Example Format:
{{
    "bed_0": {{
        "parent": "floor_0",
        "isAgainstWall": true,
        "directlyFacing": null,
    }},
    "TV_0": {{
        "parent": "TV_stand_0",
        "isAgainstWall": false,
        "directlyFacing": "tabel_0",
    }}
    "chandelier_0": {{
        "parent": "ceiling_0",
        "isAgainstWall": false,
    }},
    ...
}}

Remember, any object not listed in **items_in_region** ({items_in_region}) should not be included in the scene graph generation process.
'''

GENERATE_SCENE_GRAPH_PROMPT_FLOOR_WALL="""
Task Overview:
Create a scene graph for the objects identified in pic_1 <image-placeholder>, 
specifically those within the designated region, referred to as **items_in_region**: {items_in_region}.

Reference Images:
pic_2 <image-placeholder>: The complete scene image, listing all objects, is referred to as **all_items_list**: {all_items_list}.
{wall_color_name}

Object Attributes:
For each object in pic_1, populate the following attributes:
1.isAgainstWall: Determine if the object is directly against a wall, specifically with its back touching the wall. This means the object is placed in such a way that its rear surface is aligned with or adjacent to the wall. If it is, set this to true; otherwise, set it to false.

2.isOnFloor: Determine if the object is directly on the floor. This means the base of the object is resting on the ground surface without any elevation. If it is, set this to true; otherwise, set it to false.

3.isHangingFromCeiling: Determine if the object is hanging from the ceiling. This implies the object is suspended from above, without any support from below. If it is, set this to true; otherwise, set it to false.

4.isHangingOnWall: Determine if the object is hanging on the wall. This indicates the object is affixed to the wall, typically using hooks or nails, without resting on any horizontal surface. If it is, set this to true; otherwise, set it to false.

Follow the steps below to complete the task:
step1:Identify the object's isAgainstWall attribute and give the reason.
step2:Identify the object's isOnFloor attribute and give the reason.
step3:Identify the object's isHangingFromCeiling attribute and give the reason.
step4:Identify the object's isHangingOnWall attribute and give the reason.
step5:Output the result in the following format:

Example Format:
{{
    "bed_0": {{
        "isAgainstWall": true,
        "isOnFloor": true,
        "isHangingFromCeiling": false,
        "isHangingOnWall": false,
    }},
    "TV_0": {{
        "isAgainstWall": true,
        "isOnFloor": false,
        "isHangingFromCeiling": false,
        "isHangingOnWall": false,
    }}
    "chandelier_0": {{
        "isAgainstWall": false,
        "isOnFloor": false,
        "isHangingFromCeiling": true,
        "isHangingOnWall": false,
    }},
    ...
}}

Remember, any object not listed in **items_in_region** ({items_in_region}) should not be included in the scene graph generation process.
"""


'''
You should return every step.
'''


GENERATE_SEMANTIC_RELATIONSHIPS_PROMPT = """
You are an expert in 3D scene understanding and spatial relationships. Analyze the provided image showing objects in a room scene with their labels.

**Your Task:**
1. **Identify Visually Identical Object Groups**: Find objects that are EXACTLY the same physical item/model with identical dimensions, design, and appearance - objects that would use the exact same 3D asset without any modifications.
2. **Identify Surrounding/Facing Relationships**: ONLY focus on chairs/stools that form a group around a specific table.

**Object List in this image:** {object_list}

**Critical Grouping Rules:**
- **PERFECT VISUAL IDENTITY REQUIRED** - Objects must be:
  - **EXACTLY the same physical item/model** - not just similar, but identical instances
  - **EXACTLY the same real-world dimensions** as they appear in the image
  - **EXACTLY the same height, width, and depth** - zero size variation allowed
  - **EXACTLY the same proportions, scale, design, color, material, and every visual detail**

- **SIZE MATCHING IS MANDATORY:**
  - **Compare objects pixel by pixel** - if one looks even slightly larger or smaller, DO NOT group
  - **Account for perspective** - objects further away may look smaller but must be the same real size
  - **ZERO tolerance for size differences** - even the tiniest size difference means NO grouping

- **EXCEPTION FOR SMALL OBJECTS**: For small items like cups, books, desk accessories, decorative items - allow minor size/detail variations if they are very similar in appearance and function.

- **ULTRA-CONSERVATIVE GROUPING:**
  - **These must be identical copies/instances of the same exact item**
  - **DO NOT group objects just because they are the same category or type**
  - **DO NOT group similar-looking objects** - they must be absolutely identical
  - **Minimum group size is 3 objects**
  - **When in doubt, DO NOT group** - be extremely conservative
  - **If you see ANY difference in size, shape, design, or appearance - DO NOT group**
  - **For paintings or pictures, the image content inside the frame must be identical; do not group them if only the frames match.**

**Facing Relationship Rules (SIMPLIFIED):**
- **ONLY consider chairs/stools groups that surround/face a specific table**
- The chairs/stools must be visually identical AND collectively surround one table
- Ignore other types of facing relationships

Return a python Dict in the following format:
{{
  "groups": {{
    "group_0": ["chair_1", "chair_2", "chair_5"],
    "group_1": ["monitor_1", "monitor_2"],
    "group_2": ["chair_3", "chair_4"],
  }},
  "facing_relationships": {{
    "group_0": "dining_table_1",
  }}
}}
"""

FLOOR_VERIFICATION_PROMPT = """You are an expert in 3D scene understanding.
Your task is to verify whether objects are actually supported by the floor.

**Input:**
A grid image containing cropped views of several objects. Each object's name is labeled in its respective grid cell.

**Instructions:**
1. Carefully examine each object in the grid.
2. The red bounding box in each image indicates the complete object to be evaluated. Treat everything within the red box as a single, unified object (e.g., a potted plant includes both the plant and its container/vase as one object).
3. Determine what surface is supporting the bottom of this complete object.
4. Objects that are clearly supported by furniture (tables, shelves, beds, cabinets, etc.) or hanging on walls should be marked as NOT supported by the floor.
5. Only mark an object as floor-supported if the bottom of the object (or its base/container) is directly touching the floor.
6. Objects placed on thin floor coverings (such as rugs, carpets, or mats) should still be considered as supported by the floor.
7. For each object, return True if it IS supported by the floor, False if it is NOT supported by the floor.

**Objects to verify:**
{object_names}

Now, analyze the provided image and return your assessment in the specified JSON format (JSON only):
```json
{{
  "object_name": {{
    "reason": "Brief explanation of whether the object is supported by the floor",
    "is_floor_supported": true  // or false
  }}
}}
"""


VLM_SIZE_CORRECTION_PROMPT = """
You are an expert in 3D scene understanding.
Your task is to refine the estimated dimensions (length, width, height) of objects based on their cropped images.
The provided dimensions represent the size of each object's 3D bounding box and are derived from noisy and potentially occluded point cloud data.

**Definitions & Coordinate System:**
- All dimensions refer to the object's 3D bounding box (not the object's intrinsic dimensions)
- The coordinate system is aligned with the camera's perspective:
  - **Length (X-axis)**: Horizontal dimension in the camera view. Generally accurate.
  - **Width (Y-axis)**: Depth dimension (distance from camera). **This is typically the LEAST accurate and MOST likely to need correction.**
  - **Height (Z-axis)**: Vertical dimension. **This is absolutely correct and should be trusted as a firm reference.**

**Input:**
1. A grid image containing cropped views of several objects. Each object's name is labeled in its respective grid cell.
2. A list of the objects shown in the grid, along with their initial, uncorrected dimensions in centimeters [length, width, height].

**Instructions:**
1. Carefully examine the cropped image for each object.
2. Use the provided `height` as a reliable anchor point. Do not change the height.
3. Assess the visual proportions of each object relative to its height.
4. **Focus primarily on correcting the `width` (depth), as this is the most error-prone dimension.**
5. Correct `length` and `width` only when they are clearly inconsistent with the visual evidence. If the original values appear reasonable, retain them.
6. Ensure your corrected dimensions are plausible for the object type and consistent with its appearance.
7. Provide corrected dimensions for ALL objects listed below, even if no changes are needed.

**Objects and Initial Dimensions (in cm), ordered as [Length, Width, Height]:**
{initial_dimensions}

Now, analyze the provided image and return the refined dimensions in the specified JSON format(JSON only):
```json
{{
  "object_name":  [corrected_length_cm, corrected_width_cm, height],
}}
```
"""
