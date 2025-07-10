# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'ElfinAndError'
copyright = '2024, firstelfin'
author = 'firstelfin'

release = '0.1.1'
version = '0.1.1'

# -- General configuration

extensions = [
    'myst_parser',
    'sphinxcontrib.mermaid',
    'sphinx_copybutton',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinxcontrib.htmlhelp',
    'sphinxcontrib.serializinghtml',  # 支持多格式输出
    'sphinx.ext.imgmath',
    'sphinx_markdown_tables',
]

# 可选配置
imgmath_image_format = 'svg'  # 或 png
imgmath_font_size = 14
# imgmath_latex_preamble = r'\usepackage{amsmath}'
# imgmath_latex_preamble = r'''
#     \usepackage{xcolor}
#     \usepackage{amsmath}
# '''


source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

language = 'zh_CN'
html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Options for MyST
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",  # 允许 $ 作为数学公式的开始符号
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",  # 允许使用 ! 替换图片链接
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_fence_as_directive = ["mermaid"]  # 允许 mermaid 作为一个指令, 没有这一行，前面的扩展不会生效

latex_engine = 'xelatex'
