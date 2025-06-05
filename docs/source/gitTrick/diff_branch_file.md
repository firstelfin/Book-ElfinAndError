# Git文件、分支更新

## 1、不同分支差异查询

查询两个分支差异：

```shell
$ git diff branch1 branch2

diff --git a/sphinx_study/sphinx_utils/add2.py b/sphinx_study/sphinx_utils/add2.py
index bef1078..f27bbd3 100644
--- a/sphinx_study/sphinx_utils/add2.py
+++ b/sphinx_study/sphinx_utils/add2.py
@@ -40,3 +40,8 @@ class SuppressParam(BaseModel):
     pass
+
+
+def add_test():
+    print("-*-"*12)
+
diff --git a/sphinx_study/sphinx_utils/repo_dir/sphinx_add.py b/sphinx_study/sphinx_utils/repo_dir/sphinx_add.py
index 50423d6..0697250 100644
--- a/sphinx_study/sphinx_utils/repo_dir/sphinx_add.py
+++ b/sphinx_study/sphinx_utils/repo_dir/sphinx_add.py
@@ -41,3 +41,8 @@ class SuppressParam(BaseModel):
     pass
+
+
+def test_elfin():
+    print("-*-"*12)
+
```

查询不同分支文件的差异：

```shell
$ git diff branch1 branch2 test.py

index 1ca3046..f27bbd3 100644
--- a/sphinx_study/sphinx_utils/add2.py
+++ b/sphinx_study/sphinx_utils/add2.py
@@ -42,3 +42,6 @@ class SuppressParam(BaseModel):
     pass
 
 
+def add_test():
+    print("-*-"*12)
+
```

## 2、不同commit差异查询

## 3、不同分支文件替换

不同分支文件合并（merge要合并全部文件）：

```shell
$ git checkout -p test sphinx_study/sphinx_utils/add2.py
iff --git b/sphinx_study/sphinx_utils/add2.py a/sphinx_study/sphinx_utils/add2.py
index 1ca3046..f27bbd3 100644
--- b/sphinx_study/sphinx_utils/add2.py
+++ a/sphinx_study/sphinx_utils/add2.py
@@ -42,3 +42,6 @@ class SuppressParam(BaseModel):
     pass
 
 
+def add_test():
+    print("-*-"*12)
+
(1/1) Apply this hunk to index and worktree [y,n,q,a,d,e,?]? y
<stdin>:11: new blank line at EOF.
+
warning: 1 行新增了空白字符误用。
<stdin>:11: new blank line at EOF.
+
warning: 1 行新增了空白字符误用。
```

如果不小心多合并了代码，则需要进行回退：

```shell
$ git reset commit_id
$ git restore xxx.py
```