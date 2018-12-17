import React from 'react';
import { ScrollView, FlatList, Text, StyleSheet, Button, View, SectionList } from 'react-native';

export default class TestScreen extends React.Component {

  static navigationOptions = {
    title: 'Test',
  };


  render() {

    console.log("What I Got!");
    const params = this.props.navigation.state.params.params;

    return (
        <FlatList style={styles.flatList} data={params.list}
          renderItem={({item, index, section}) =>
            <View style={styles.questionView} key={index}>
              <Text style={styles.questionText}>{item.question}</Text>
            </View>}
        />
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'column',
    justifyContent: 'center',
    backgroundColor: 'red',
  },
  flatList: {
    flex: 1,
    flexDirection: 'column',
    backgroundColor: 'green',
  },
  flatListContent: {
    flexDirection: 'column',
    backgroundColor: 'yellow',
    justifyContent: 'center',
    alignItems: 'center',
  },
  questionText: {
    flex: 1,
    flexDirection: 'row',
  },
  questionView: {
    flex: 1,
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'orange',
  },
});
