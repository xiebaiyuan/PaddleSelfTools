 #!/bin/bash
 function ergodic(){
#     echo "$1"
     for file in ` ls -a $1 `
     do
	 if [ $file == . ] || [ $file == .. ] || [ $file == ".DS_Store" ]
         then
             continue
         fi
         if [ -d $1/$file ]
         then
             ergodic $1/$file
         else
             md5sum $1/$file | sed s#$INIT_PATH/## >> $RECORDFILE
         fi
     done
 }

 #获取当前目录
 INIT_PATH="$1"
 cd "$INIT_PATH"
 #设置输出文件名
 RECORDFILE=all.md5
 #如果存在先删除，防止重复运行脚本时追加到记录文件
 test -e $RECORDFILE && rm $RECORDFILE
 #遍历所有文件
 ergodic $INIT_PATH

 #删除不需要的隐藏文件
# sed -i / \./d $RECORDFILE
 #按文件名称排序
 #sort t -k 2 $RECORDFILE -o $RECORDFILE

