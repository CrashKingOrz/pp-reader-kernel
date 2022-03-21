CREATE DATABASE IF NOT EXISTS bysms default charset utf8 COLLATE utf8_general_ci;

use bysms;

DROP TABLE IF EXISTS `user_table`;

CREATE TABLE `user_table`(
	`id` int(11) unsigned NOT NULL AUTO_INCREMENT,
	`open_id` varchar(100) DEFAULT NULL COMMENT '微信open_id',
	`name` varchar(100) DEFAULT NULL COMMENT '姓名',
	`gender` varchar(100) DEFAULT NULL COMMENT '性别',
	`phone_num` varchar(100) DEFAULT NULL COMMENT '电话',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `user_table` (`id`, `open_id`, `name`, `gender`, `phone_num`)
VALUES
(1, 'ABCDEFG', 'xdx', '男', '13322334455');


DROP TABLE IF EXISTS `query_table`;

CREATE TABLE `query_table`(
	`id` int(11) unsigned NOT NULL AUTO_INCREMENT,
	`open_id` varchar(100) DEFAULT NULL COMMENT '用户open_id',
    `type` varchar(100) DEFAULT NULL COMMENT '查询类型',
    `content` varchar(100) DEFAULT NULL COMMENT '查询内容',
    `time` DATETIME DEFAULT NULL COMMENT '查询时间',
    `result` varchar(1000) COMMENT '查询结果',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `query_table` (`id`, `open_id`, `type`, `content`, `time`, `result`)
VALUES
(1, 'ABCDEFG', '英语单词', 'tiger', '2022-03-18 09:30:27', '{"ret": 0,"relist": {"例子": [{"英": "The tiger is native to India.","汉": "这种虎产于印度。"},{"英": "Which has spots, a leopard or a tiger?","汉": "有斑点的是豹还是虎？"},{"英": "Which has spots, the leopard or the tiger?","汉": "有斑点的是豹还是虎？"}],"英标": ["[ˈtaɪɡə(r)]","[ˈtaɪɡər]"],"释义": ["n. 老虎；凶暴的人","n. （Tiger）人名；（英）泰格；（法）蒂热；（瑞典）蒂格"]}}');


DROP TABLE IF EXISTS `time_table`;

CREATE TABLE `time_table`(
	`id` int(11) unsigned NOT NULL AUTO_INCREMENT,
    `open_id` varchar(100) DEFAULT NULL COMMENT '用户open_id',
	`start_time` DATETIME NOT NULL COMMENT '学习开始时间',
    `end_time` DATETIME DEFAULT NULL COMMENT '学习结束时间',
    `duration` int(11) unsigned DEFAULT 0 COMMENT '学习时长(s)',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO `time_table` (`id`, `open_id`, `start_time`, `end_time`, `duration`)
VALUES
(1, 'ABCDEFG', '2022-03-18 09:33:30', '2022-03-18 09:34:30', 60);
